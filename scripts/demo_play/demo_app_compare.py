#!/usr/bin/env python3
import os
import re
import json
import shutil
import zipfile
import threading
from pathlib import Path
from flask import Flask, send_file, request, jsonify, abort, Response, render_template_string
import cv2
import numpy as np

# ================== 配置区（请按需修改） ==================
DATA_FOLDER_A = "/vla/users/luojingzhou/data/checkpoints/GR00TN-1.6-InterVL3.5-3B/liberoplus/liberoplus_all_6w_pgd_step1_wonoiseadv_epsilon0d03_cslearning_tasklist/checkpoint-60000/eval_liberoplus/all/rollout_video"
DATA_FOLDER_B = "/vla/users/luojingzhou/data/checkpoints/GR00TN-1.6-InterVL3.5-3B/liberoplus/liberoplus_all-6w-tasklist/checkpoint-60000/eval_liberoplus/all/rollout_video"  # 示例路径，请替换

BASE_DIR_A = Path(DATA_FOLDER_A)
BASE_DIR_B = Path(DATA_FOLDER_B)
CACHE_DIR = Path('/vla/users/luojingzhou/data/checkpoints/GR00TN-1.6-InterVL3.5-3B/demo_cache')
CACHE_DIR.mkdir(exist_ok=True)
PAGE_SIZE = 16  # 每页任务数（注意：每个任务含 A+B 两视频）
QUALITY = 95    # JPEG质量
# =========================================================


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB limit

# 全局任务注册表: {task_name: {"A": {...}, "B": {...}}}
task_registry = {}
registry_lock = threading.Lock()
scan_completed = False
scan_lock = threading.Lock()

# ================== CSS 样式（内联） ==================
CSS_STYLE = """
body { 
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
    margin: 0; 
    background: #f5f7fa; 
    color: #333;
}
.container { 
    max-width: 1800px; 
    margin: 0 auto; 
    padding: 20px;
}
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px 30px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 600;
}
.header .subtitle {
    opacity: 0.9;
    margin-top: 5px;
    font-size: 15px;
}
.filters { 
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    align-items: center;
}
.filter-group {
    display: flex;
    flex-direction: column;
    min-width: 200px;
}
.filter-label {
    font-size: 13px;
    color: #666;
    margin-bottom: 5px;
    font-weight: 500;
}
#keyword-input { 
    padding: 10px 12px;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 14px;
    width: 100%;
    box-sizing: border-box;
    transition: border-color 0.2s;
}
#keyword-input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}
.status-filter {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.status-btn {
    padding: 8px 16px;
    border: 2px solid #e0e0e0;
    background: white;
    border-radius: 20px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
}
.status-btn:hover {
    border-color: #667eea;
    background: #f8f9ff;
}
.status-btn.active {
    border-color: #667eea;
    background: #667eea;
    color: white;
}
.status-btn.success { border-color: #4CAF50; }
.status-btn.success:hover { border-color: #45a049; background: #f1f9f1; }
.status-btn.success.active { border-color: #45a049; background: #4CAF50; }
.status-btn.fail { border-color: #f44336; }
.status-btn.fail:hover { border-color: #e53935; background: #fff1f0; }
.status-btn.fail.active { border-color: #e53935; background: #f44336; color: white; }
.status-btn.compare { border-color: #ff9800; }
.status-btn.compare:hover { border-color: #f57c00; background: #fff8e1; }
.status-btn.compare.active { border-color: #f57c00; background: #ff9800; color: white; }

.filter-actions {
    display: flex;
    gap: 10px;
    margin-left: auto;
}
.btn-filter {
    padding: 10px 20px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s;
}
.btn-filter:hover {
    background: #5568d3;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
}

.stats-bar {
    background: white;
    padding: 12px 20px;
    border-radius: 8px;
    margin-bottom: 25px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.stats-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 500;
}
.stats-item.success { color: #4CAF50; }
.stats-item.fail { color: #f44336; }
.stats-item.total { color: #555; font-weight: 600; }
.stats-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}
.stats-dot.success { background: #4CAF50; }
.stats-dot.fail { background: #f44336; }
.stats-dot.total { background: #2196F3; }

.grid { 
    display: grid; 
    grid-template-columns: repeat(auto-fill, minmax(750px, 1fr)); 
    gap: 25px;
}
.task-row {
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.task-header {
    padding: 12px 16px;
    background: #f8f9fa;
    border-bottom: 1px solid #eee;
    font-weight: 600;
    font-size: 14px;
    word-break: break-all;
    min-height: 40px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.videos-container {
    display: flex;
    gap: 1px;
    background: #eee;
}
.video-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: white;
    padding: 12px;
}
.panel-title {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
    text-align: center;
    color: #555;
}
.status-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    z-index: 10;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.video-container { 
    position: relative;
    width: 100%; 
    height: 180px; 
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    display: flex; 
    align-items: center; 
    justify-content: center;
    overflow: hidden;
    border-radius: 6px;
    margin-bottom: 10px;
}
video { 
    max-width: 96%; 
    max-height: 160px; 
    outline: none;
    border-radius: 4px;
}
.btn-group { 
    display: flex; 
    justify-content: center; 
    gap: 6px; 
    flex-wrap: wrap;
}
.btn { 
    padding: 5px 10px; 
    font-size: 11px; 
    border: none; 
    border-radius: 4px; 
    cursor: pointer; 
    transition: all 0.2s;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 4px;
}
.btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.btn:active {
    transform: translateY(0);
}
.btn-download { 
    background: #4CAF50; 
    color: white;
}
.btn-download:hover { 
    background: #45a049;
}
.btn-frames { 
    background: #2196F3; 
    color: white;
}
.btn-frames:hover { 
    background: #1976d2;
}
.btn-generating { 
    background: #ff9800; 
    color: white; 
    cursor: not-allowed;
}
.btn-generating:hover {
    background: #fb8c00;
    transform: none;
    box-shadow: none;
}

.pagination { 
    display: flex; 
    justify-content: center; 
    margin-top: 30px; 
    gap: 8px;
    flex-wrap: wrap;
}
.page-info {
    margin: 0 20px;
    color: #666;
    font-size: 14px;
    align-self: center;
}
.page-btn { 
    padding: 8px 16px; 
    border: 1px solid #ddd; 
    background: white; 
    cursor: pointer; 
    border-radius: 6px;
    font-size: 14px;
    transition: all 0.2s;
    min-width: 40px;
}
.page-btn:hover {
    border-color: #667eea;
    background: #f8f9ff;
}
.page-btn.active { 
    background: #667eea; 
    color: white; 
    border-color: #667eea;
    font-weight: 600;
}
.page-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.tip-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid #2196F3;
    padding: 20px;
    border-radius: 8px;
    margin-top: 30px;
    font-size: 14px;
    line-height: 1.6;
}
.tip-box h3 {
    margin: 0 0 10px 0;
    color: #1565c0;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.tip-box ul {
    margin: 10px 0 0 20px;
    padding-left: 0;
}
.tip-box li {
    margin-bottom: 6px;
    color: #424242;
}
.tip-box code {
    background: rgba(255,255,255,0.7);
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
}
"""

# ================== HTML 模板（内联） ==================
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rollout Video Viewer - A/B 对比</title>
    <style>
    {{ css_style|safe }}
    </style>
    <script>
    function filterTasks() {
        const kw = document.getElementById('keyword-input').value;
        const status = document.querySelector('.status-btn.active').dataset.status;
        const url = new URL(window.location);
        url.searchParams.set('keywords', kw);
        url.searchParams.set('status', status);
        url.searchParams.set('page', '1');
        window.location.href = url.toString();
    }
    
    function setStatusFilter(status) {
        document.querySelectorAll('.status-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
        filterTasks();
    }
    
    function checkFrameStatus(taskName, side, btn) {
        fetch(`/api/check_zip/${taskName}/${side}`)
        .then(r => r.json())
        .then(data => {
            if (data.ready) {
                btn.className = 'btn btn-frames';
                btn.onclick = () => window.location.href=`/download/frames/${taskName}/${side}`;
                btn.innerHTML = '🖼️ 帧序列';
                btn.disabled = false;
            } else {
                setTimeout(() => checkFrameStatus(taskName, side, btn), 2000);
            }
        });
    }
    
    function triggerFrameDownload(taskName, side, btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ 生成中...';
        fetch(`/download/frames/${taskName}/${side}`)
        .then(r => {
            if (r.status === 202) {
                setTimeout(() => checkFrameStatus(taskName, side, btn), 1500);
            } else if (r.ok) {
                return r.blob();
            } else {
                throw new Error('生成失败');
            }
        })
        .then(blob => {
            if (blob) {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${taskName}_${side}_frames.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }
        })
        .catch(e => {
            alert('下载失败: ' + e.message);
            btn.disabled = false;
            btn.innerHTML = '🖼️ 帧序列';
        });
    }
    </script>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🎬 Rollout Video Viewer - A/B 对比</h1>
        <div class="subtitle">论文素材整理工具 · 支持 A/B 结果对比（如：消融实验）</div>
    </div>
    
    <div class="stats-bar">
        <div class="stats-item total">
            <span class="stats-dot total"></span>
            共 <strong>{{ total }}</strong> 个任务
        </div>
        {% if status_filter == 'a_success_b_fail' %}
        <div class="stats-item success">
            <span class="stats-dot success"></span>
            A 成功
        </div>
        <div class="stats-item fail">
            <span class="stats-dot fail"></span>
            B 失败
        </div>
        {% endif %}
    </div>
    
    <div class="filters">
        <div class="filter-group">
            <div class="filter-label">🔍 关键词过滤</div>
            <input type="text" id="keyword-input" 
                   placeholder="输入关键词（逗号分隔，如：kitchen,stove）" 
                   value="{{ keywords }}"
                   onkeydown="if(event.key==='Enter') filterTasks()">
        </div>
        
        <div class="filter-group">
            <div class="filter-label">📊 状态筛选</div>
            <div class="status-filter">
                <button class="status-btn {% if status_filter == 'all' %}active{% endif %}" 
                        data-status="all" onclick="setStatusFilter('all')">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#2196F3;margin-right:4px;"></span>
                    全部
                </button>
                <button class="status-btn compare {% if status_filter == 'a_success_b_fail' %}active{% endif %}" 
                        data-status="a_success_b_fail" onclick="setStatusFilter('a_success_b_fail')">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ff9800;margin-right:4px;"></span>
                    A成功 & B失败
                </button>
            </div>
        </div>
        
        <div class="filter-actions">
            <button class="btn-filter" onclick="filterTasks()">🔄 应用过滤</button>
        </div>
    </div>
    
    <div class="grid">
    {% for task in tasks %}
        <div class="task-row">
            <div class="task-header" title="{{ task.name }}">{{ task.name }}</div>
            <div class="videos-container">
                <!-- A Panel -->
                <div class="video-panel">
                    <div class="panel-title">📁 A ({{ DATA_FOLDER_A_NAME }})</div>
                    <div class="video-container">
                        {% if task.A %}
                        <div class="status-badge" style="background:{{ task.A.status_color }}">
                            {{ task.A.status_text }}
                        </div>
                        <video controls preload="metadata" muted>
                            <source src="/video/{{ task.name }}/A" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        {% else %}
                        <div style="color:white;font-size:14px;">❌ 无视频</div>
                        {% endif %}
                    </div>
                    <div class="btn-group">
                        {% if task.A %}
                        <button class="btn btn-download" 
                                onclick="window.location.href='/download/video/{{ task.name }}/A'">
                            📥 视频
                        </button>
                        {% if task.A.zip_ready %}
                        <button class="btn btn-frames" 
                                onclick="window.location.href='/download/frames/{{ task.name }}/A'">
                            🖼️ 帧序列
                        </button>
                        {% else %}
                        <button class="btn btn-frames" 
                                onclick="triggerFrameDownload('{{ task.name }}', 'A', this)">
                            🖼️ 帧序列
                        </button>
                        {% endif %}
                        {% endif %}
                    </div>
                </div>
                
                <!-- B Panel -->
                <div class="video-panel">
                    <div class="panel-title">📁 B ({{ DATA_FOLDER_B_NAME }})</div>
                    <div class="video-container">
                        {% if task.B %}
                        <div class="status-badge" style="background:{{ task.B.status_color }}">
                            {{ task.B.status_text }}
                        </div>
                        <video controls preload="metadata" muted>
                            <source src="/video/{{ task.name }}/B" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        {% else %}
                        <div style="color:white;font-size:14px;">❌ 无视频</div>
                        {% endif %}
                    </div>
                    <div class="btn-group">
                        {% if task.B %}
                        <button class="btn btn-download" 
                                onclick="window.location.href='/download/video/{{ task.name }}/B'">
                            📥 视频
                        </button>
                        {% if task.B.zip_ready %}
                        <button class="btn btn-frames" 
                                onclick="window.location.href='/download/frames/{{ task.name }}/B'">
                            🖼️ 帧序列
                        </button>
                        {% else %}
                        <button class="btn btn-frames" 
                                onclick="triggerFrameDownload('{{ task.name }}', 'B', this)">
                            🖼️ 帧序列
                        </button>
                        {% endif %}
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    {% endfor %}
    </div>
    
    {% if total_pages > 1 %}
    <div class="pagination">
        <button class="page-btn {% if page == 1 %}disabled{% endif %}" 
                {% if page > 1 %}onclick="window.location.href='/?keywords={{ keywords|urlencode }}&status={{ status_filter }}&page={{ page-1 }}'"{% endif %}>
            ←
        </button>
        
        {% set start_page = [1, page - 2]|max %}
        {% set end_page = [total_pages, page + 2]|min %}
        
        {% if start_page > 1 %}
        <button class="page-btn" onclick="window.location.href='/?keywords={{ keywords|urlencode }}&status={{ status_filter }}&page=1'">1</button>
        {% if start_page > 2 %}
        <span style="align-self:center">...</span>
        {% endif %}
        {% endif %}
        
        {% for p in range(start_page, end_page + 1) %}
        <button class="page-btn {% if p == page %}active{% endif %}" 
                onclick="window.location.href='/?keywords={{ keywords|urlencode }}&status={{ status_filter }}&page={{ p }}'">
            {{ p }}
        </button>
        {% endfor %}
        
        {% if end_page < total_pages %}
        {% if end_page < total_pages - 1 %}
        <span style="align-self:center">...</span>
        {% endif %}
        <button class="page-btn" onclick="window.location.href='/?keywords={{ keywords|urlencode }}&status={{ status_filter }}&page={{ total_pages }}'">
            {{ total_pages }}
        </button>
        {% endif %}
        
        <button class="page-btn {% if page == total_pages %}disabled{% endif %}" 
                {% if page < total_pages %}onclick="window.location.href='/?keywords={{ keywords|urlencode }}&status={{ status_filter }}&page={{ page+1 }}'"{% endif %}>
            →
        </button>
        
        <div class="page-info">
            第 {{ page }} / {{ total_pages }} 页
        </div>
    </div>
    {% endif %}
    
    <div class="tip-box">
        <h3>💡 论文使用建议（A/B 对比）</h3>
        <ul>
            <li><strong>A/B 对比</strong>：点击“A成功 & B失败”快速定位消融实验中的关键案例</li>
            <li><strong>帧序列</strong>：ZIP 包含高质量 JPG 帧和 <code>metadata.json</code>，可直接用于论文插图</li>
            <li><strong>命名规范</strong>：A/B 文件夹路径可在代码顶部配置，建议命名为有意义的实验名</li>
            <li><strong>批量分析</strong>：结合关键词（如 <code>open_door</code>）筛选特定场景下的 A/B 表现差异</li>
        </ul>
    </div>
</div>
</body>
</html>
"""

# ================== 辅助函数 ==================
def parse_status_from_name(name: str):
    """从文件名判断成功/失败状态"""
    if name.endswith('_1') or (name[-1] == '1' and not name.endswith('_0')):
        return "success", "✅ 成功", "#4CAF50"
    elif name.endswith('_0') or (name[-1] == '0' and not name.endswith('_1')):
        return "fail", "❌ 失败", "#f44336"
    else:
        return "unknown", "❓ 未知", "#9e9e9e"

def scan_tasks():
    """扫描 A 和 B 目录，构建联合任务注册表"""
    global task_registry, scan_completed
    with scan_lock:
        if scan_completed:
            return
        
        registry = {}
        
        # 获取所有任务名（取 A 和 B 的并集）
        task_names = set()
        if BASE_DIR_A.exists():
            task_names.update([p.name for p in BASE_DIR_A.iterdir() if p.is_dir()])
        if BASE_DIR_B.exists():
            task_names.update([p.name for p in BASE_DIR_B.iterdir() if p.is_dir()])
        
        for task_name in sorted(task_names):
            entry = {"A": None, "B": None}
            
            # 处理 A
            if BASE_DIR_A.exists():
                a_dir = BASE_DIR_A / task_name
                if a_dir.is_dir():
                    mp4s = list(a_dir.glob("*.mp4"))
                    if mp4s:
                        main_vid = None
                        for vid in mp4s:
                            status, text, color = parse_status_from_name(vid.stem)
                            if status == "success":
                                main_vid = vid
                                break
                        if not main_vid:
                            main_vid = mp4s[0]
                            status, text, color = parse_status_from_name(main_vid.stem)
                        
                        zip_path = CACHE_DIR / f"{task_name}_A.zip"
                        entry["A"] = {
                            "video_path": main_vid,
                            "status": status,
                            "status_text": text,
                            "status_color": color,
                            "zip_ready": zip_path.exists(),
                            "zip_path": zip_path,
                            "all_videos": mp4s
                        }
            
            # 处理 B
            if BASE_DIR_B.exists():
                b_dir = BASE_DIR_B / task_name
                if b_dir.is_dir():
                    mp4s = list(b_dir.glob("*.mp4"))
                    if mp4s:
                        main_vid = None
                        for vid in mp4s:
                            status, text, color = parse_status_from_name(vid.stem)
                            if status == "success":
                                main_vid = vid
                                break
                        if not main_vid:
                            main_vid = mp4s[0]
                            status, text, color = parse_status_from_name(main_vid.stem)
                        
                        zip_path = CACHE_DIR / f"{task_name}_B.zip"
                        entry["B"] = {
                            "video_path": main_vid,
                            "status": status,
                            "status_text": text,
                            "status_color": color,
                            "zip_ready": zip_path.exists(),
                            "zip_path": zip_path,
                            "all_videos": mp4s
                        }
            
            # 只保留至少有一边有视频的任务
            if entry["A"] or entry["B"]:
                registry[task_name] = entry
        
        with registry_lock:
            task_registry = registry
        
        # 统计
        total = len(registry)
        a_success_b_fail = 0
        for t in registry.values():
            a_ok = t["A"] and t["A"]["status"] == "success"
            b_fail = t["B"] and t["B"]["status"] == "fail"
            if a_ok and b_fail:
                a_success_b_fail += 1
        
        print(f"✅ 扫描完成：共 {total} 个任务")
        print(f"   - A成功且B失败: {a_success_b_fail}")
        scan_completed = True

def ensure_scanned():
    if not scan_completed:
        threading.Thread(target=scan_tasks, daemon=True).start()

def generate_frames_zip(task_name, side):
    """生成指定 side (A/B) 的帧 ZIP"""
    with registry_lock:
        if task_name not in task_registry or not task_registry[task_name].get(side):
            return
        entry = task_registry[task_name][side]
        if entry["zip_ready"]:
            return
    
    vid_path = entry["video_path"]
    zip_path = CACHE_DIR / f"{task_name}_{side}.zip"
    temp_dir = CACHE_DIR / f"_temp_{task_name}_{side}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise ValueError("无法打开视频")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            out_path = frames_dir / f"frame_{i+1:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
        cap.release()
        
        meta = {
            "task_name": task_name,
            "side": side,
            "video_file": vid_path.name,
            "status": entry["status"],
            "status_text": entry["status_text"],
            "total_frames": frame_count,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "generated_by": "Rollout Viewer A/B Compare"
        }
        with open(temp_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(temp_dir))
        
        with registry_lock:
            if task_name in task_registry and side in task_registry[task_name]:
                task_registry[task_name][side]["zip_ready"] = True
                task_registry[task_name][side]["zip_path"] = zip_path
        
        print(f"✅ 帧包生成完成: {task_name} ({side})")
    
    except Exception as e:
        print(f"❌ 生成帧包失败 {task_name} ({side}): {str(e)}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

# ================== 路由 ==================
@app.route('/')
def index():
    ensure_scanned()
    
    keywords = request.args.get('keywords', '').strip()
    status_filter = request.args.get('status', 'all')
    page = max(1, int(request.args.get('page', 1)))
    
    with registry_lock:
        tasks = list(task_registry.keys())
    
    # 关键词过滤
    if keywords:
        kw_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
        if kw_list:
            tasks = [t for t in tasks if all(kw in t.lower() for kw in kw_list)]
    
    # 状态过滤
    filtered_tasks = []
    with registry_lock:
        for t in tasks:
            entry = task_registry[t]
            if status_filter == 'a_success_b_fail':
                a_ok = entry["A"] and entry["A"]["status"] == "success"
                b_fail = entry["B"] and entry["B"]["status"] == "fail"
                if a_ok and b_fail:
                    filtered_tasks.append(t)
            else:  # 'all'
                filtered_tasks.append(t)
    
    filtered_tasks.sort()
    total = len(filtered_tasks)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_tasks = filtered_tasks[start:end]
    
    task_data = []
    with registry_lock:
        for name in page_tasks:
            task_data.append({
                "name": name,
                "A": task_registry[name].get("A"),
                "B": task_registry[name].get("B")
            })
    
    return render_template_string(
        INDEX_TEMPLATE,
        css_style=CSS_STYLE,
        tasks=task_data,
        keywords=keywords,
        status_filter=status_filter,
        page=page,
        total_pages=total_pages,
        total=total,
        DATA_FOLDER_A_NAME="A",
        DATA_FOLDER_B_NAME="B"
    )

@app.route('/video/<task_name>/<side>')
def stream_video(task_name, side):
    ensure_scanned()
    with registry_lock:
        if task_name not in task_registry or not task_registry[task_name].get(side):
            abort(404)
        video_path = task_registry[task_name][side]["video_path"]
    
    if not video_path.exists():
        abort(404)
    
    def generate():
        with open(video_path, 'rb') as f:
            while chunk := f.read(8192):
                yield chunk
    return Response(generate(), mimetype='video/mp4')

@app.route('/download/video/<task_name>/<side>')
def download_video(task_name, side):
    ensure_scanned()
    with registry_lock:
        if task_name not in task_registry or not task_registry[task_name].get(side):
            abort(404)
        video_path = task_registry[task_name][side]["video_path"]
    
    if not video_path.exists():
        abort(404)
    status = task_registry[task_name][side]["status"]
    return send_file(video_path, as_attachment=True, download_name=f"{task_name}_{side}_{status}.mp4")

@app.route('/download/frames/<task_name>/<side>')
def download_frames(task_name, side):
    ensure_scanned()
    with registry_lock:
        if task_name not in task_registry or not task_registry[task_name].get(side):
            abort(404)
        need_generate = not task_registry[task_name][side]["zip_ready"]
    
    if need_generate:
        threading.Thread(target=generate_frames_zip, args=(task_name, side), daemon=True).start()
        return jsonify({"status": "generating"}), 202
    
    with registry_lock:
        zip_path = task_registry[task_name][side]["zip_path"]
    
    if not zip_path.exists():
        return jsonify({"error": "帧包不存在"}), 404
    return send_file(zip_path, as_attachment=True, download_name=f"{task_name}_{side}_frames.zip")

@app.route('/api/check_zip/<task_name>/<side>')
def check_zip_status(task_name, side):
    ensure_scanned()
    with registry_lock:
        ready = task_registry.get(task_name, {}).get(side, {}).get("zip_ready", False)
    return jsonify({"ready": ready})

# ================== 启动 ==================
if __name__ == '__main__':
    print("="*70)
    print("🚀 Rollout Video Viewer (A/B 对比版) 启动中...")
    print(f"📁 A 目录: {BASE_DIR_A}")
    print(f"📁 B 目录: {BASE_DIR_B}")
    print(f"💾 缓存目录: {CACHE_DIR}")
    print("✅ 功能亮点：")
    print("   • 支持 A/B 两组 rollout 视频对比")
    print("   • 一键筛选「A成功 & B失败」的关键案例")
    print("   • 左右分栏展示，方便视觉对比")
    print("   • 帧序列含 metadata，论文引用超方便")
    print("="*70)
    
    threading.Thread(target=scan_tasks, daemon=True).start()
    app.run(host='0.0.0.0', port=47653, debug=False, threaded=True)
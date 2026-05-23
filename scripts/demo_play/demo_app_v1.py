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
data_folder = "/data2/luojingzhou/datasets/checkpoints/GR00TN-1.6-InterVL3.5-3B/liberoplus/liberoplus_all_6w_pgd_step1_wonoiseadv_epsilon0d03_cslearning_tasklist/checkpoint-60000/eval_liberoplus/all/rollout_video"
BASE_DIR = Path(data_folder)
CACHE_DIR = Path('/data2/luojingzhou/datasets/checkpoints/GR00TN-1.6-InterVL3.5-3B/demo_cache')
CACHE_DIR.mkdir(exist_ok=True)
PAGE_SIZE = 16  # 4x4网格
QUALITY = 95    # JPEG质量（论文推荐90-95）
# =========================================================


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB limit

# 全局任务注册表
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
    max-width: 1600px; 
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
.status-btn.unknown { border-color: #9e9e9e; }
.status-btn.unknown:hover { border-color: #757575; background: #f5f5f5; }
.status-btn.unknown.active { border-color: #757575; background: #9e9e9e; color: white; }

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
    grid-template-columns: repeat(4, 1fr); 
    gap: 20px;
}
.card { 
    background: white; 
    border-radius: 10px; 
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    display: flex;
    flex-direction: column;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
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
    height: 200px; 
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    display: flex; 
    align-items: center; 
    justify-content: center;
    overflow: hidden;
}
video { 
    max-width: 96%; 
    max-height: 180px; 
    outline: none;
    border-radius: 4px;
}
.card-footer { 
    padding: 12px; 
    text-align: center; 
    font-size: 12.5px; 
    word-break: break-all;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}
.task-name {
    font-weight: 500;
    color: #333;
    margin-bottom: 8px;
    line-height: 1.4;
    min-height: 40px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.btn-group { 
    display: flex; 
    justify-content: center; 
    gap: 8px; 
    margin-top: 8px;
    flex-wrap: wrap;
}
.btn { 
    padding: 6px 12px; 
    font-size: 12px; 
    border: none; 
    border-radius: 5px; 
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
.btn-all { 
    background: #9c27b0; 
    color: white;
}
.btn-all:hover { 
    background: #7b1fa2;
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
    <title>Rollout Video Viewer - 论文素材整理</title>
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
    
    function checkFrameStatus(taskName, btn) {
        fetch(`/api/check_zip/${taskName}`)
        .then(r => r.json())
        .then(data => {
            if (data.ready) {
                btn.className = 'btn btn-frames';
                btn.onclick = () => window.location.href=`/download/frames/${taskName}`;
                btn.innerHTML = '🖼️ 帧序列';
                btn.disabled = false;
            } else {
                setTimeout(() => checkFrameStatus(taskName, btn), 2000);
            }
        });
    }
    
    function triggerFrameDownload(taskName, btn) {
        btn.disabled = true;
        btn.innerHTML = '⏳ 生成中...';
        fetch(`/download/frames/${taskName}`)
        .then(r => {
            if (r.status === 202) {
                setTimeout(() => checkFrameStatus(taskName, btn), 1500);
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
                a.download = `${taskName}_frames.zip`;
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
        <h1>🎬 Rollout Video Viewer</h1>
        <div class="subtitle">论文素材整理工具 · 快速筛选成功/失败演示</div>
    </div>
    
    <div class="stats-bar">
        <div class="stats-item total">
            <span class="stats-dot total"></span>
            共 <strong>{{ total }}</strong> 个任务
        </div>
        <div class="stats-item success">
            <span class="stats-dot success"></span>
            <strong>{{ success_count }}</strong> 个成功
        </div>
        <div class="stats-item fail">
            <span class="stats-dot fail"></span>
            <strong>{{ fail_count }}</strong> 个失败
        </div>
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
                <button class="status-btn success {% if status_filter == 'success' %}active{% endif %}" 
                        data-status="success" onclick="setStatusFilter('success')">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#4CAF50;margin-right:4px;"></span>
                    仅成功
                </button>
                <button class="status-btn fail {% if status_filter == 'fail' %}active{% endif %}" 
                        data-status="fail" onclick="setStatusFilter('fail')">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f44336;margin-right:4px;"></span>
                    仅失败
                </button>
            </div>
        </div>
        
        <div class="filter-actions">
            <button class="btn-filter" onclick="filterTasks()">🔄 应用过滤</button>
        </div>
    </div>
    
    <div class="grid">
    {% for task in tasks %}
        <div class="card">
            <div class="video-container">
                <div class="status-badge" style="background:{{ task.status_color }}">
                    {{ task.status_text }}
                </div>
                <video controls preload="metadata" muted>
                    <source src="/video/{{ task.name }}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            <div class="card-footer">
                <div class="task-name" title="{{ task.name }}">{{ task.name }}</div>
                <div class="btn-group">
                    <button class="btn btn-download" title="下载主视频" 
                            onclick="window.location.href='/download/video/{{ task.name }}'">
                        📥 视频
                    </button>
                    {% if task.zip_ready %}
                    <button class="btn btn-frames" title="下载帧序列ZIP" 
                            onclick="window.location.href='/download/frames/{{ task.name }}'">
                        🖼️ 帧序列
                    </button>
                    {% else %}
                    <button class="btn btn-frames" title="首次点击会生成帧序列" 
                            onclick="triggerFrameDownload('{{ task.name }}', this)">
                        🖼️ 帧序列
                    </button>
                    {% endif %}
                    {% if task.has_multiple_videos %}
                    <button class="btn btn-all" title="下载该任务所有视频（成功+失败）" 
                            onclick="window.location.href='/download/all_videos/{{ task.name }}'">
                        📦 全部
                    </button>
                    {% endif %}
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
        <h3>💡 论文使用建议</h3>
        <ul>
            <li><strong>状态筛选</strong>：点击"仅成功"快速定位可用演示，"仅失败"分析失败案例</li>
            <li><strong>帧序列下载</strong>：ZIP包含 <code>frames/frame_00001.jpg</code> 高质量图片，可直接插入LaTeX/Overleaf</li>
            <li><strong>metadata.json</strong>：记录分辨率、帧数、成功/失败状态，方便在论文中说明实验设置</li>
            <li><strong>批量下载</strong>：点击"📦 全部"下载同一任务的成功+失败视频，便于对比分析</li>
            <li><strong>关键词过滤</strong>：支持多关键词（如 <code>stove,coffee</code>），快速定位特定场景</li>
        </ul>
    </div>
</div>
</body>
</html>
"""

# ================== 核心功能 ==================
def scan_tasks():
    """扫描BASE_DIR，注册有效任务（含mp4文件），识别成功/失败状态"""
    global task_registry, scan_completed
    with scan_lock:
        if scan_completed:
            return
        
        new_registry = {}
        if not BASE_DIR.exists():
            print(f"⚠️ 警告：BASE_DIR不存在: {BASE_DIR}")
            scan_completed = True
            return
        
        for item in BASE_DIR.iterdir():
            if not item.is_dir():
                continue
            
            # 查找所有mp4文件
            mp4s = list(item.glob("*.mp4"))
            if not mp4s:
                continue
            
            # 识别成功/失败状态（支持 *_1.mp4 和 *1.mp4 两种命名）
            success_video = None
            fail_video = None
            other_videos = []
            
            for vid in mp4s:
                name = vid.stem  # 不含扩展名
                if name.endswith('_1') or (name[-1] == '1' and not name.endswith('_0')):
                    success_video = vid
                elif name.endswith('_0') or (name[-1] == '0' and not name.endswith('_1')):
                    fail_video = vid
                else:
                    other_videos.append(vid)
            
            # 确定主视频和状态
            if success_video:
                main_video = success_video
                status = "success"
                status_text = "✅ 成功"
                status_color = "#4CAF50"
            elif fail_video:
                main_video = fail_video
                status = "fail"
                status_text = "❌ 失败"
                status_color = "#f44336"
            elif other_videos:
                main_video = other_videos[0]
                status = "unknown"
                status_text = "❓ 未知"
                status_color = "#9e9e9e"
            else:
                continue
            
            zip_path = CACHE_DIR / f"{item.name}.zip"
            new_registry[item.name] = {
                "video_path": main_video,
                "status": status,
                "status_text": status_text,
                "status_color": status_color,
                "zip_ready": zip_path.exists(),
                "zip_path": zip_path,
                "all_videos": mp4s,
                "has_multiple_videos": len(mp4s) > 1
            }
        
        with registry_lock:
            task_registry = new_registry
        
        # 统计信息
        success_count = sum(1 for t in task_registry.values() if t["status"] == "success")
        fail_count = sum(1 for t in task_registry.values() if t["status"] == "fail")
        unknown_count = sum(1 for t in task_registry.values() if t["status"] == "unknown")
        
        print(f"✅ 扫描完成：共 {len(task_registry)} 个有效任务")
        print(f"   - 成功: {success_count} | 失败: {fail_count} | 未知: {unknown_count}")
        scan_completed = True

def ensure_scanned():
    """确保任务已扫描（懒加载）"""
    if not scan_completed:
        threading.Thread(target=scan_tasks, daemon=True).start()

def generate_frames_zip(task_name):
    """后台生成帧ZIP（含metadata）"""
    with registry_lock:
        if task_name not in task_registry or task_registry[task_name]["zip_ready"]:
            return
        entry = task_registry[task_name]
    
    vid_path = entry["video_path"]
    zip_path = CACHE_DIR / f"{task_name}.zip"
    temp_dir = CACHE_DIR / f"_temp_{task_name}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise ValueError("无法打开视频")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 保存帧
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            out_path = frames_dir / f"frame_{i+1:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
        cap.release()
        
        # 生成metadata
        meta = {
            "task_name": task_name,
            "video_file": vid_path.name,
            "status": entry["status"],
            "status_text": entry["status_text"],
            "total_frames": frame_count,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "generated_by": "Rollout Viewer for Paper",
            "note": "Frames named as frame_00001.jpg for easy inclusion in LaTeX/Overleaf"
        }
        with open(temp_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        # 打包
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(temp_dir))
        
        # 更新注册表
        with registry_lock:
            if task_name in task_registry:
                task_registry[task_name]["zip_ready"] = True
                task_registry[task_name]["zip_path"] = zip_path
        
        print(f"✅ 帧包生成完成: {task_name} ({frame_count} frames, {entry['status_text']})")
    
    except Exception as e:
        print(f"❌ 生成帧包失败 {task_name}: {str(e)}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

# ================== 路由 ==================
@app.route('/')
def index():
    ensure_scanned()  # 确保已扫描
    
    keywords = request.args.get('keywords', '').strip()
    status_filter = request.args.get('status', 'all')
    page = max(1, int(request.args.get('page', 1)))
    
    # 过滤任务
    with registry_lock:
        tasks = list(task_registry.keys())
    
    # 关键词过滤
    if keywords:
        kw_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
        if kw_list:
            tasks = [t for t in tasks if all(kw in t.lower() for kw in kw_list)]
    
    # 状态过滤
    if status_filter != 'all':
        with registry_lock:
            tasks = [t for t in tasks if task_registry.get(t, {}).get('status') == status_filter]
    
    # 排序
    tasks.sort()
    
    # 分页
    total = len(tasks)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_tasks = tasks[start:end]
    
    # 准备任务数据
    task_data = []
    with registry_lock:
        for name in page_tasks:
            if name in task_registry:
                task_data.append({
                    "name": name,
                    "status": task_registry[name]["status"],
                    "status_text": task_registry[name]["status_text"],
                    "status_color": task_registry[name]["status_color"],
                    "zip_ready": task_registry[name]["zip_ready"],
                    "has_multiple_videos": task_registry[name]["has_multiple_videos"]
                })
    
    # 统计
    success_count = sum(1 for t in task_data if t["status"] == "success")
    fail_count = sum(1 for t in task_data if t["status"] == "fail")
    
    return render_template_string(
        INDEX_TEMPLATE,
        css_style=CSS_STYLE,
        tasks=task_data,
        keywords=keywords,
        status_filter=status_filter,
        page=page,
        total_pages=total_pages,
        total=total,
        success_count=success_count,
        fail_count=fail_count
    )

@app.route('/video/<task_name>')
def stream_video(task_name):
    ensure_scanned()
    with registry_lock:
        if task_name not in task_registry:
            abort(404)
        video_path = task_registry[task_name]["video_path"]
    
    if not video_path.exists():
        abort(404)
    
    def generate():
        with open(video_path, 'rb') as f:
            f.seek(0)
            while chunk := f.read(8192):
                yield chunk
    return Response(generate(), mimetype='video/mp4')

@app.route('/download/video/<task_name>')
def download_video(task_name):
    ensure_scanned()
    with registry_lock:
        if task_name not in task_registry:
            abort(404)
        video_path = task_registry[task_name]["video_path"]
    
    if not video_path.exists():
        abort(404)
    return send_file(video_path, as_attachment=True, download_name=f"{task_name}_{task_registry[task_name]['status']}.mp4")

@app.route('/download/all_videos/<task_name>')
def download_all_videos(task_name):
    ensure_scanned()
    with registry_lock:
        if task_name not in task_registry:
            abort(404)
        all_videos = task_registry[task_name].get("all_videos", [])
    
    if not all_videos:
        abort(404)
    
    import tempfile
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_zip.close()
    
    try:
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for vid_path in all_videos:
                zf.write(vid_path, vid_path.name)
        
        return send_file(
            temp_zip.name,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{task_name}_all_videos.zip",
            etag=False
        )
    finally:
        if os.path.exists(temp_zip.name):
            os.unlink(temp_zip.name)

@app.route('/download/frames/<task_name>')
def download_frames(task_name):
    ensure_scanned()
    with registry_lock:
        if task_name not in task_registry:
            abort(404)
        need_generate = not task_registry[task_name]["zip_ready"]
    
    if need_generate:
        threading.Thread(target=generate_frames_zip, args=(task_name,), daemon=True).start()
        return jsonify({"status": "generating", "message": "帧包生成中，请10秒后刷新页面重试下载"}), 202
    
    with registry_lock:
        zip_path = task_registry[task_name]["zip_path"]
    
    if not zip_path.exists():
        return jsonify({"error": "帧包不存在，请重试"}), 404
    return send_file(zip_path, as_attachment=True, download_name=f"{task_name}_frames.zip")

@app.route('/api/check_zip/<task_name>')
def check_zip_status(task_name):
    ensure_scanned()
    with registry_lock:
        ready = task_registry.get(task_name, {}).get("zip_ready", False)
    return jsonify({"ready": ready})

# ================== 启动 ==================
if __name__ == '__main__':
    print("="*70)
    print("🚀 Rollout Video Viewer 启动中...")
    print(f"📁 视频根目录: {BASE_DIR}")
    print(f"💾 帧缓存目录: {CACHE_DIR}")
    print("✅ 功能亮点：")
    print("   • 智能识别成功/失败视频（*1.mp4 = 成功，*0.mp4 = 失败）")
    print("   • 状态筛选：快速过滤成功/失败演示")
    print("   • 帧序列ZIP：含metadata.json，论文引用超方便")
    print("   • 批量下载：一键获取同一任务的所有视频")
    print("="*70)
    
    # 启动时异步扫描（不阻塞启动）
    threading.Thread(target=scan_tasks, daemon=True).start()
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=47653, debug=False, threaded=True)
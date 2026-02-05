import os
import time
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.genai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools

# --- [1] 环境与代理配置 ---
load_dotenv()

# 从环境变量读取代理配置（如果存在）
http_proxy = os.getenv("HTTP_PROXY", "")
https_proxy = os.getenv("HTTPS_PROXY", "")
if http_proxy:
    os.environ["HTTP_PROXY"] = http_proxy
if https_proxy:
    os.environ["HTTPS_PROXY"] = https_proxy

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ 未找到 GOOGLE_API_KEY 环境变量，请在 .env 文件中配置")
    exit(1)

# --- [2] Flask 应用初始化 ---
app = Flask(__name__)
CORS(app)

# 全局变量存储视频文件ID和聊天历史
video_file_id = None
chat_history = []

# --- [3] 工具函数定义 ---
def analyze_drone_video(query: str) -> str:
    """分析无人机巡检视频内容"""
    global video_file_id
    if not video_file_id:
        return "提示：当前系统中未发现挂载的视频，请告知用户先上传视频。"

    try:
        client = genai.Client(api_key=API_KEY)
        content = [
            {"file_data": {"file_uri": f"https://generativelanguage.googleapis.com/v1beta/{video_file_id}",
                           "mime_type": "video/mp4"}},
            f"作为巡检视觉专家，请针对该视频回答以下问题：{query}"
        ]
        response = client.models.generate_content(model="gemini-2.5-flash", contents=content)
        return response.text
    except Exception as e:
        return f"视觉分析执行出错: {str(e)}"

def web_search(query: str) -> str:
    """联网搜索最新信息"""
    try:
        from duckduckgo_search import DDGS

        # 使用 DuckDuckGo 搜索
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

            if not results:
                return "未找到相关搜索结果"

            # 格式化搜索结果
            formatted_results = "🔍 搜索结果：\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. **{result['title']}**\n"
                formatted_results += f"   {result['body']}\n"
                formatted_results += f"   来源: {result['href']}\n\n"

            return formatted_results
    except Exception as e:
        return f"搜索出错: {str(e)}"

# --- [4] Agent 初始化 ---
def get_drone_agent():
    tools_list = [
        analyze_drone_video,
        web_search,  # 添加联网搜索工具
        MCPTools(command="npx -y @modelcontextprotocol/server-duckduckgo"),
        MCPTools(command="npx -y @modelcontextprotocol/server-weather")
    ]

    return Agent(
        name="智能分析助手",
        model=Gemini(id="models/gemini-2.5-flash", api_key=API_KEY),
        tools=tools_list,
        instructions=[
            "你是一个智能分析助手，具备多模态分析和信息检索能力。",
            "",
            "**核心能力：**",
            "1. 视频分析：当用户上传视频后，可以调用 analyze_drone_video 工具分析视频内容",
            "2. 实时搜索：使用 web_search 工具搜索最新法规、新闻、政策等实时信息",
            "3. 天气查询：使用 weather server 查询天气信息",
            "4. 网络搜索：使用 search server 进行其他网络搜索",
            "",
            "**特色领域：**",
            "- 擅长低空巡检、无人机监测、空域管理等专业领域",
            "- 熟悉航空法规、安全规范、应急处置等知识",
            "- 能够结合视频内容提供专业的巡检建议和风险评估",
            "",
            "**工作原则：**",
            "1. 灵活应对：不局限于巡检场景，可以回答各类问题",
            "2. 工具优先：遇到需要实时信息或视频分析的问题，主动调用相应工具",
            "3. 专业建议：在巡检、监测等专业领域，提供深度分析和决策建议",
            "4. 清晰表达：使用 **粗体** 标记重点，用 * 列出要点，保持回复结构清晰",
            "",
            "记住：你是一个全能助手，巡检只是你的专长之一，而非全部。"
        ],
        markdown=True
    )

agent = get_drone_agent()

# --- [5] 路由定义 ---
@app.route('/')
def index():
    """返回主页面"""
    return render_template('index.html')

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """处理视频上传"""
    global video_file_id

    if 'video' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    temp_path = None
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video.save(tmp.name)
            temp_path = tmp.name

        # 上传到 Gemini API
        client = genai.Client(api_key=API_KEY)
        file_ref = client.files.upload(file=temp_path)

        # 等待处理完成（最多5分钟）
        max_wait_time = 300
        start_time = time.time()

        while file_ref.state == "PROCESSING":
            if time.time() - start_time > max_wait_time:
                return jsonify({'error': '视频处理超时'}), 408
            time.sleep(2)
            file_ref = client.files.get(name=file_ref.name)

        if file_ref.state == "FAILED":
            return jsonify({'error': '视频处理失败'}), 500

        video_file_id = file_ref.name
        return jsonify({
            'success': True,
            'message': '视频上传成功',
            'file_id': video_file_id
        })

    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天消息"""
    global chat_history

    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    try:
        # 添加用户消息到历史
        chat_history.append({'role': 'user', 'content': user_message})

        # 调用 Agent 处理
        response = agent.run(user_message)
        assistant_message = response.content

        # 添加助手回复到历史
        chat_history.append({'role': 'assistant', 'content': assistant_message})

        return jsonify({
            'success': True,
            'message': assistant_message,
            'history': chat_history
        })

    except Exception as e:
        error_message = f"❌ 分析过程出错: {str(e)}"
        chat_history.append({'role': 'assistant', 'content': error_message})
        return jsonify({'error': error_message}), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    """获取聊天历史"""
    return jsonify({'history': chat_history})

if __name__ == '__main__':
    # 创建 templates 目录（如果不存在）
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("🚁 低空巡检系统启动中...")
    print("📍 访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

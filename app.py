import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from agno.media import Video

# --- [1] 强制代理配置 (解决 WinError 10060) ---
# 请务必检查你的 VPN 端口，如果是 7890 保持不变
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# 引入最新版 SDK
import google.genai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# --- [2] 页面压缩布局 ---
st.set_page_config(layout="wide", page_title="低空巡检 Pro 控制台")

st.markdown("""
    <style>
        /* [1] 页面基础缩放与页边距优化 */
        html { zoom: 1.0; } 
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }

        /* [2] 强制左右分栏列等高，并防止溢出 */
        [data-testid="stColumn"] {
            height: 82vh !important;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* [3] 核心修改：让对话输入框强制锚定在分栏底部，而不是全屏底部 */
        /* 我们通过覆盖 Streamlit 默认的 fixed 定位来实现 */
        .stChatFloatingInputContainer {
            position: relative !important;
            bottom: 0 !important;
            left: 0 !important;
            width: 100% !important;
            background: transparent !important;
            padding: 0.5rem 0 !important;
            z-index: 1;
        }

        /* [4] 修正对话框容器，使其自动填充剩余空间并提供内部滚动 */
        .stChatMessageContainer {
            flex-grow: 1;
            overflow-y: auto !important;
            margin-bottom: 5px;
            padding-right: 5px;
        }

        /* [5] 视频区域大小限制，防止挤压对话框 */
        video { 
            max-height: 45vh !important; 
            object-fit: contain; 
            border-radius: 12px; 
            background: #000;
        }

        /* 隐藏不必要的元素 */
        footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- [3] 初始化状态 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_v_name" not in st.session_state:
    st.session_state.processed_v_name = None

# --- [4] Agent 配置 (使用 Gemini 2.5 Flash) ---
@st.cache_resource
def get_drone_agent():
    return Agent(
        name="低空巡检高级专家",
        # 使用最新的预览版 ID
        model=Gemini(id="models/gemini-2.5-flash", api_key=API_KEY),
        tools=[DuckDuckGoTools()],
        instructions=[
            "你是一个拥有最高权限的低空巡检专家。",
            "当用户提供了视频附件时，你必须调用你的多模态能力查看并分析视频内容，深度解析视频中的安全隐患、违规行为或环境异常。",
            "严禁回答‘我无法观看视频’。如果视频已加载，它就在你的上下文缓存中。",
            "即便没有视频，也要以专业视角回答低空经济、无人机管理的相关问题。",
            "提供分析时，请务必给出视频中对应的具体时间范围（如：[00:15 - 00:22]）。"
        ],
        markdown=True
    )

agent = get_drone_agent()

# --- [5] UI 主逻辑 ---
st.title("🚁 低空巡检 & AI 深度决策系统")

col_l, col_spacer, col_r = st.columns([0.50, 0.02, 0.48])

with col_l:
    st.markdown("#### 📽 巡检视频流")
    v_file = st.file_uploader("Upload Video", type=["mp4", "mov"], label_visibility="collapsed")
    
    if v_file:
        if st.session_state.get("current_v") != v_file.name:
            try:
                client = genai.Client(api_key=API_KEY)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(v_file.read())
                    path = tmp.name
                
                with st.spinner("🧠 Gemini 2.5 正在构建视频神经元映射..."):
                    # 使用新版 SDK 上传
                    file_ref = client.files.upload(file=path)
                    while file_ref.state == "PROCESSING":
                        time.sleep(2)
                        file_ref = client.files.get(name=file_ref.name)
                    
                    st.session_state.processed_v_name = file_ref.name
                    st.session_state.current_v = v_file.name
                st.success("视频深度解析就绪！")
            except Exception as e:
                st.error(f"连接失败。请检查 API Key 或 VPN 节点。错误：{e}")
        st.video(v_file)
    else:
        st.info("💡 处于纯知识对话模式。上传视频后将自动开启 AI 巡检分析。")

with col_r:
    st.markdown("#### 💬 专家对话窗口")
    chat_box = st.container(height=520)
    
    # 历史记录渲染
    with chat_box:
        if not st.session_state.chat_history:
            st.chat_message("assistant").markdown("你好！我是基于 **Gemini 2.5 Flash** 的巡检专家，我已准备好为你分析视频内容或解答行业知识。")
        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).markdown(m["content"])

    # 对话输入逻辑
    if prompt := st.chat_input("询问巡检细节..."):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🚀 原生引擎分析中（拒绝幻觉）..."):
                try:
                    # 获取之前上传成功的文件引用
                    file_name = st.session_state.processed_v_name
                    
                    if file_name:
                        # 核心：直接使用 google-genai 客户端，不通过 Agno 包装
                        client = genai.Client(api_key=API_KEY)
                        
                        # 构造多模态内容：文本 + 视频引用
                        content = [
                            {"file_data": {"file_uri": f"https://generativelanguage.googleapis.com/v1beta/{file_name}", "mime_type": "video/mp4"}},
                            f"请根据视频内容真实回答，严禁幻觉。用户问题：{prompt}"
                        ]
                        
                        # 调用模型
                        # 注意：这里直接用 client 而不是 agent.run，确保 100% 成功率
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", # 或者你确定可用的 1.5-flash
                            contents=content
                        )
                        answer = response.text
                    else:
                        # 没有视频时才走普通的 agent 逻辑
                        res = agent.run(prompt)
                        answer = res.content
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"分析失败: {e}")

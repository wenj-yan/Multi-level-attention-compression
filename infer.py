"""
Qwen Streamlit Web Demo - 基于 chat_infer.py
"""
import re
import sys
import os
import time
import queue
from threading import Thread
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from compression_chat_session import ChatSession

st.set_page_config(page_title="Chat", initial_sidebar_state="collapsed")

# 初始化 session_state（必须在最前面）
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_messages = []
if "qwen_session" not in st.session_state:
    st.session_state.qwen_session = None

# 初始化配置参数（这些是可配置的）
if "compress_mode" not in st.session_state:
    st.session_state.compress_mode = "avg"
if "compress_layers" not in st.session_state:
    st.session_state.compress_layers = "4"
if "compress_strides" not in st.session_state:
    st.session_state.compress_strides = "4,4"
if "level_caps" not in st.session_state:
    st.session_state.level_caps = "1024,1024"
if "mem_len" not in st.session_state:
    st.session_state.mem_len = "512"  # 增加到600以更好地展示压缩过程
if "debug_compression" not in st.session_state:
    st.session_state.debug_compression = False
if "config_saved" not in st.session_state:
    st.session_state.config_saved = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "config_locked" not in st.session_state:
    st.session_state.config_locked = False

# 自定义CSS：减少侧边栏padding + 创建右侧固定统计面板（响应式布局）
st.markdown("""
<style>
    /* 左侧配置栏样式 */
    [data-testid="stSidebarUserContent"]  {
        padding-top: 0.5rem !important;
        margin-top: 0.5rem !important;
    }
    
    /* 右侧统计面板容器 - 响应式设计 + 毛玻璃效果 */
    .stats-sidebar {
        position: fixed;
        top: 60px;
        right: 20px;
        width: 320px;
        max-width: 25vw;  /* 最大宽度为视口宽度的25% */
        max-height: calc(100vh - 80px);
        overflow-y: auto;
        
        /* 毛玻璃效果 */
        background: rgba(248, 249, 250, 0.75);  /* 半透明背景 */
        backdrop-filter: blur(10px);  /* 背景模糊 */
        -webkit-backdrop-filter: blur(10px);  /* Safari支持 */
        
        border: 1px solid rgba(255, 255, 255, 0.3);  /* 半透明边框 */
        border-radius: 15px;  /* 更大的圆角 */
        padding: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);  /* 更柔和的阴影 */
        z-index: 999;
        transition: width 0.3s ease;  /* 平滑过渡效果 */
    }
    
    /* 主内容区域留出右侧空间 - 响应式 */
    .main .block-container {
        max-width: calc(100% - 360px);
        margin-right: max(360px, 26vw);  /* 使用视口宽度的26%或360px中较大的值 */
        transition: margin-right 0.3s ease;  /* 平滑过渡效果 */
    }
    
    /* 统计面板标题 */
    .stats-sidebar h3 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 18px;
        color: #333;
    }
    
    /* 统计面板滚动条美化 */
    .stats-sidebar::-webkit-scrollbar {
        width: 6px;
    }
    
    .stats-sidebar::-webkit-scrollbar-thumb {
        background-color: #ccc;
        border-radius: 3px;
    }
    
    .stats-sidebar::-webkit-scrollbar-track {
        background-color: #f1f1f1;
    }
    
    /* 中等屏幕（平板） */
    @media screen and (max-width: 1400px) {
        .stats-sidebar {
            width: 280px;
            max-width: 28vw;
        }
        .main .block-container {
            margin-right: max(300px, 30vw);
        }
    }
    
    /* 小屏幕（窄屏） */
    @media screen and (max-width: 1024px) {
        .stats-sidebar {
            width: 240px;
            max-width: 30vw;
            padding: 10px;
        }
        .main .block-container {
            margin-right: max(260px, 32vw);
        }
        .stats-sidebar h3 {
            font-size: 16px;
        }
    }
    
    /* 超小屏幕（移动设备） - 隐藏右侧面板 */
    @media screen and (max-width: 768px) {
        .stats-sidebar {
            display: none;  /* 在小屏幕上隐藏统计面板 */
        }
        .main .block-container {
            margin-right: 0;
            max-width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)


# 模型配置
st.sidebar.subheader("模型路径")
# 下拉框选择模型
model_options = [
    "qwen/Qwen2-7B-Instruct",
    "qwen/Qwen2-14B-Instruct",
    "qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B-Chat"
]
qwen_model_name = st.sidebar.selectbox(
    "模型名称", 
    options=model_options,
    index=0,
    disabled=st.session_state.config_locked
)

qwen_local_dir = "./models/qwen2-7b-instruct/Qwen/Qwen2-7B-Instruct"
# qwen_local_dir = st.sidebar.text_input("本地路径", value="./qwen2-7b-instruct/qwen/qwen2-7b-instruct")

# 压缩配置
st.sidebar.subheader("KV Cache 压缩配置")
st.session_state.mem_len = st.sidebar.text_input(
    "记忆长度 (mem_len)", 
    value=str(st.session_state.mem_len),
    disabled=st.session_state.config_locked
)

# 将compress_strides拆分为两个滑动条
col1, col2 = st.sidebar.columns(2)
with col1:
    if isinstance(st.session_state.compress_strides, str):
        stride_values = list(map(int, st.session_state.compress_strides.split(",")))
    else:
        stride_values = [4, 4]
    stride_l1 = st.slider(
        "压缩率L1", 
        min_value=1, 
        max_value=10, 
        value=stride_values[0],
        disabled=st.session_state.config_locked,
        help="一级压缩步长"
    )
with col2:
    stride_l2 = st.slider(
        "压缩率L2", 
        min_value=1, 
        max_value=10, 
        value=stride_values[1] if len(stride_values) > 1 else 4,
        disabled=st.session_state.config_locked,
        help="二级压缩步长"
    )
# 合并回字符串格式
st.session_state.compress_strides = f"{stride_l1},{stride_l2}"

# 将level_caps拆分为两个输入框
col3, col4 = st.sidebar.columns(2)
with col3:
    if isinstance(st.session_state.level_caps, str):
        cap_values = list(map(int, st.session_state.level_caps.split(",")))
    else:
        cap_values = [1024, 1024]
    cap_l1 = st.text_input(
        "压缩容量L1", 
        value=str(cap_values[0]),
        disabled=st.session_state.config_locked,
        help="一级压缩缓存容量（tokens）"
    )
with col4:
    cap_l2 = st.text_input(
        "压缩容量L2", 
        value=str(cap_values[1]) if len(cap_values) > 1 else "1024",
        disabled=st.session_state.config_locked,
        help="二级压缩缓存容量（tokens）"
    )
# 合并回字符串格式
st.session_state.level_caps = f"{cap_l1},{cap_l2}"
st.session_state.compress_mode = st.sidebar.selectbox(
    "压缩模式 (compress_mode)", 
    options=["avg", "mlp"],
    index=0 if st.session_state.compress_mode == "avg" else 1,
    help="avg=平均池化(无需训练), mlp=可训练MLP(需要load_weights)",
    disabled=st.session_state.config_locked
)
st.session_state.compress_layers = st.sidebar.text_input(
    "压缩层配置 (compress_layers)", 
    value=st.session_state.compress_layers,
    help="数字N=后N层(如4=后4层,16=后16层), all=全部层, 或指定层号如'24,25,26,27'",
    disabled=st.session_state.config_locked
)



# 保存配置按钮
st.sidebar.markdown("---")
if not st.session_state.config_locked:
    if st.sidebar.button("💾 保存配置并加载模型", use_container_width=True, type="primary"):
        st.session_state.config_saved = True
        st.session_state.config_locked = True
        st.session_state.model_loaded = False
        # 清除旧的session实例，强制重新创建
        st.session_state.qwen_session = None
        # 重置对话历史
        st.session_state.messages = []
        st.session_state.chat_messages = []
        # 立即rerun以应用disabled状态
        st.rerun()
else:
    #  (相当于/reset)
    if st.sidebar.button("🔄 重置配置", use_container_width=True, type="secondary"):
        st.session_state.config_saved = False
        st.session_state.config_locked = False
        st.session_state.model_loaded = False
        # 清除session实例
        st.session_state.qwen_session = None
        # 重置对话历史
        st.session_state.messages = []
        st.session_state.chat_messages = []
        st.rerun()

if st.session_state.config_saved:
    if st.session_state.qwen_session is None and st.session_state.config_locked:
        st.sidebar.warning("⏳ 正在保存配置...")
    else:
        st.sidebar.success("✅ 配置已保存并锁定")
else:
    st.sidebar.warning("⚠️ 请先保存配置")
    
# 调试选项
st.sidebar.markdown("---")
st.sidebar.subheader("调试选项")
st.session_state.debug_compression = st.sidebar.checkbox(
    "启用压缩调试 (debug_compression)", 
    value=st.session_state.debug_compression,
    help="在控制台打印详细的KV cache压缩统计信息",
    disabled=st.session_state.config_locked
)

# 页面标题和样式（使用简单文字标题以避免 emoji 在某些环境中显示问题）
st.markdown(
    '<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    '<span style="font-size: 26px;">💬 Chat (Streaming)</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">流式对话</span>'
    '</div>',
    unsafe_allow_html=True
)

def display_compression_stats(placeholder, qwen_session):
    """在右侧面板显示KV Cache压缩统计"""
    if qwen_session is None:
        with placeholder.container():
            st.markdown("""
            <div class="stats-sidebar">
                <h3>📊 实时统计</h3>
                <p style="color: #666; font-size: 14px;">等待模型加载...</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    stats = qwen_session.get_compression_stats()
    if stats is None:
        with placeholder.container():
            st.markdown("""
            <div class="stats-sidebar">
                <h3>📊 实时统计</h3>
                <p style="color: #666; font-size: 14px;">获取统计信息中...</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # 准备数据
    mem_usage = stats['mem_tokens']
    mem_cap = stats['mem_cap']
    mem_pct = min(100, (mem_usage / mem_cap * 100) if mem_cap > 0 else 0)
    
    l1_usage = stats['l1_tokens']
    l1_cap = stats['l1_cap']
    l1_pct = min(100, (l1_usage / l1_cap * 100) if l1_cap > 0 else 0)
    
    l2_usage = stats['l2_tokens']
    l2_cap = stats['l2_cap']
    l2_pct = min(100, (l2_usage / l2_cap * 100) if l2_cap > 0 else 0)
    
    total = stats['total_tokens']
    original = stats['original_tokens']
    
    # 生成进度条HTML
    num_segments = 20
    
    # 原始区进度条（绿色）
    mem_filled = int((mem_pct / 100) * num_segments)
    mem_bar = ""
    for i in range(num_segments):
        if i < mem_filled:
            hue, sat, light = 120, 60 + (i * 2), 65 - (i * 2)
            mem_bar += f'<div style="width: {100/num_segments}%; height: 100%; background: hsl({hue}, {sat}%, {light}%); display: inline-block;"></div>'
        else:
            mem_bar += f'<div style="width: {100/num_segments}%; height: 100%; background: #e0e0e0; display: inline-block;"></div>'
    
    # L1进度条（黄色）
    l1_filled = int((l1_pct / 100) * num_segments)
    l1_bar = ""
    for i in range(num_segments):
        if i < l1_filled:
            l1_bar += f'<div style="width: {100/num_segments}%; height: 100%; background: hsl({60 - i * 2}, 90%, {60 - i}%); display: inline-block;"></div>'
        else:
            l1_bar += f'<div style="width: {100/num_segments}%; height: 100%; background: #e0e0e0; display: inline-block;"></div>'
    
    # L2进度条（红色）
    l2_filled = int((l2_pct / 100) * num_segments)
    l2_bar = ""
    for i in range(num_segments):
        if i < l2_filled:
            l2_bar += f'<div style="width: {100/num_segments}%; height: 100%; background: hsl(0, {90 - i * 2}%, {55 - i}%); display: inline-block;"></div>'
        else:
            l2_bar += f'<div style="width: {100/num_segments}%; height: 100%; background: #e0e0e0; display: inline-block;"></div>'
    
    # 压缩效果
    if total > 0 and original > total:
        compression_ratio = original / total
        
		# 节省的tokens数！！！
        saved = original  - total 
  
        compression_html = f"""
            <div style="margin-bottom: 12px;">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 6px;">压缩层效果：</div>
                <div style="display: flex; justify-content: space-between; font-size: 16px;">
                    <div>
                        <div style="color: #666; font-size: 16px;">压缩前</div>
                        <div style="font-weight: bold; font-size: 16px;">{original}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #666; font-size: 16px;">压缩后</div>
                        <div style="font-weight: bold; font-size: 16px;">{total}</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 6px; font-size: 16px; color: #28a745; font-weight: bold;">
                    节省 {saved} tokens ({compression_ratio:.2f}x)
                </div>
            </div>"""
    else:
        compression_html = ""
    
    # 组合完整HTML
    html_content = f"""
<div class="stats-sidebar">
    <h3 style="margin-top: 0; margin-bottom: 12px; font-size: 20px; color: #333;">📊 KV Cache 统计</h3>
    
    <div style="margin-bottom: 12px;">
        <div style="font-size: 16px; font-weight: bold; margin-bottom: 6px;">容量配置：</div>
        <div style="display: flex; justify-content: space-between; font-size: 16px;">
            <div><span style="color: #666;">原始:</span> <b>{mem_cap}</b></div>
            <div><span style="color: #666;">L1:</span> <b>{l1_cap}</b></div>
            <div><span style="color: #666;">L2:</span> <b>{l2_cap}</b></div>
        </div>
    </div>
    
    <div style="margin-bottom: 12px;">
        <div style="font-size: 16px; font-weight: bold; margin-bottom: 8px;">当前使用量：</div>
        
        <div style="margin-bottom: 8px;">
            <div style="font-size: 16px; margin-bottom: 4px;">🟢 原始: <b>{mem_usage}/{mem_cap}</b> ({mem_pct:.0f}%)</div>
            <div style="width: 100%; height: 10px; display: flex; gap: 1px; border-radius: 3px; overflow: hidden;">
                {mem_bar}
            </div>
        </div>
        
        <div style="margin-bottom: 8px;">
            <div style="font-size: 16px; margin-bottom: 4px;">🟡 L1: <b>{l1_usage}/{l1_cap}</b> ({l1_pct:.0f}%)</div>
            <div style="width: 100%; height: 10px; display: flex; gap: 1px; border-radius: 3px; overflow: hidden;">
                {l1_bar}
            </div>
        </div>
        
        <div style="margin-bottom: 8px;">
            <div style="font-size: 16px; margin-bottom: 4px;">🔴 L2: <b>{l2_usage}/{l2_cap}</b> ({l2_pct:.0f}%)</div>
            <div style="width: 100%; height: 10px; display: flex; gap: 1px; border-radius: 3px; overflow: hidden;">
                {l2_bar}
            </div>
        </div>
    </div>
    
    <div style="margin-bottom: 12px;">
        <div style="font-size: 16px; font-weight: bold; margin-bottom: 6px;">压缩事件：</div>
        <div style="display: flex; justify-content: space-between; font-size: 16px;">
            <div>
                <div style="color: #666; font-size: 16px;">L1次数</div>
                <div style="font-weight: bold;">{stats['l1_compress_events']}</div>
            </div>
            <div style="text-align: right;">
                <div style="color: #666; font-size: 16px;">L2次数</div>
                <div style="font-weight: bold;">{stats['l2_compress_events']}</div>
            </div>
        </div>
    </div>
    
    {compression_html}
</div>
"""
    
    # Use st.html() for proper HTML rendering
    placeholder.html(html_content)

# 创建右侧固定统计面板占位符
stats_placeholder = st.empty()

def load_qwen_session(model_name, local_dir, mem_len, compress_strides, level_caps, compress_mode, compress_layers, debug_compression):
    """
    创建ChatSession实例，使用用户配置的参数
    每次保存配置后都会重新创建，确保使用最新参数
    
    注意：temperature, max_new_tokens, history_chat_num 等参数使用 ChatSession 的默认值
    """
    try:
        # 确保mem_len是整数
        mem_len = int(mem_len) if isinstance(mem_len, str) else mem_len
        
        # 解析压缩步长和容量（确保都转换为整数）
        strides = tuple(map(int, compress_strides.split(",")))
        # 清理level_caps输入并转换为整数数组
        caps_str = level_caps.strip()
        caps = tuple(map(int, caps_str.split(",")))
        
        # 解析压缩层配置
        if compress_layers == "all":
            parsed_compress_layers = "all"
        elif compress_layers.isdigit():
            parsed_compress_layers = int(compress_layers)
        elif "," in compress_layers:
            parsed_compress_layers = list(map(int, compress_layers.split(",")))
        else:
            parsed_compress_layers = int(compress_layers) if compress_layers else 4
        
        # 显示解析后的配置（调试用）
        print("\n" + "="*60)
        print("【Web前端配置】传递给 ChatSession:")
        print(f"  - mem_len: {mem_len}")
        print(f"  - compress_strides: {strides}")
        print(f"  - level_caps: {caps}")
        print(f"  - compress_mode: {compress_mode.upper()}")
        print(f"  - compress_layers: {parsed_compress_layers}")
        print(f"  - debug_compression: {debug_compression}")
        print("="*60 + "\n")
        
        # 显示当前配置到侧边栏
        debug_status = "✅ 已启用" if debug_compression else "❌ 已禁用"
        st.sidebar.info(f"📋 当前配置:\n- mem_len: {mem_len}\n- strides: {strides}\n- caps: {caps}\n- mode: {compress_mode}\n- layers: {compress_layers}\n- debug: {debug_status}")
        
        # 使用 ChatSession 的默认参数（temperature=0.8, max_new_tokens=4096 等）
        session = ChatSession(
            model_name=model_name,
            local_dir=local_dir,
            mem_len=mem_len,
            compress_strides=strides,
            level_caps=caps,
            # temperature, top_p, max_new_tokens, min_new_tokens 使用默认值
            compress_mode=compress_mode,
            compress_layers=parsed_compress_layers,
            debug_compression=debug_compression,  # 传递调试选项
            debug_interval=128  # 每128个token打印一次统计
        )
        
        # 验证参数是否正确传入
        print("\n" + "="*60)
        print("【ChatSession 接收确认】:")
        print(f"  - session.mem_len: {session.mem_len}")
        print(f"  - session.compress_strides: {session.compress_strides}")
        print(f"  - session.level_caps: {session.level_caps}")
        print(f"  - session.compress_mode: {session.compress_mode.upper()}")
        print(f"  - session.compress_layers: {session.compress_layers}")
        print(f"  - session.debug_compression: {session.debug_compression}")
        print("="*60 + "\n")
        
        if debug_compression:
            print("⚠️  调试模式已启用，控制台将显示详细的压缩统计信息\n")
        
        return session
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# st.markdown(
#     '<div style="margin-top: 10px; margin-bottom: 10px; padding: 10px; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9;"'
# 	'<span style="font-size: 26px;">aaa1</span>'
# 	,unsafe_allow_html=True
# )
# st.container(border=True,width="stretch", height="content", horizontal=False, horizontal_alignment="left", vertical_alignment="top", gap="small")


def main():
    # 只在配置保存后才加载模型
    if not st.session_state.config_saved:
        st.info("👈 请在侧边栏配置参数并点击'保存配置并加载模型'按钮")
        return
    
    # 如果配置已保存但session未创建，则创建新session
    if st.session_state.qwen_session is None:
        # 在主内容区域显示加载中
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.info("🔄 正在加载模型，请稍候...")
        
        with st.spinner("正在加载模型（使用最新配置）..."):
            st.session_state.qwen_session = load_qwen_session(
                qwen_model_name, 
                qwen_local_dir, 
                st.session_state.mem_len,
                st.session_state.compress_strides,
                st.session_state.level_caps,
                st.session_state.compress_mode,
                st.session_state.compress_layers,
                st.session_state.debug_compression
            )
            if st.session_state.qwen_session is not None:
                st.session_state.model_loaded = True
                # 清除加载提示，显示成功消息
                loading_placeholder.empty()
                st.success("✅ 模型加载成功！可以开始对话了")
                print("\n已进入多轮对话模式。随时可以在Web界面输入消息。\n")
            else:
                loading_placeholder.empty()
                st.error("❌ 模型加载失败，请检查配置")
                return
    
    qwen_session = st.session_state.qwen_session
    
    # 初始显示统计信息（模型加载后）
    display_compression_stats(stats_placeholder, qwen_session)
    
    # 聊天消息显示区域
    col_chat = st.container()
    
    # 在左侧列显示历史消息
    with col_chat:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
            else:
                # 用户消息靠右显示
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: gray; border-radius: 10px; color:white;">{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
    
	
    
    # 输入框
    prompt = st.chat_input("给 Chat 发送消息")
    
    if prompt:
        # 打印用户输入到控制台
        print(f"\nuser> {prompt}\n")
        
        # 在左侧列显示用户消息
        with col_chat:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: gray; border-radius: 10px; color:white;">{prompt}</div></div>',
                unsafe_allow_html=True
            )
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # 在左侧列显示助手回复
        with col_chat:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                
                if qwen_session is None:
                    answer = "❌ 模型未加载"
                    placeholder.markdown(answer)
                else:
                    # 直接使用当前消息（不使用历史轮数限制）
                    qwen_session.messages = []
                    for msg in st.session_state.chat_messages:
                        qwen_session.messages.append(msg)
                    
                    # 打印assistant开始生成
                    print("assistant> ", end="", flush=True)
                    
                    # 根据调试模式选择输出方式
                    if qwen_session.debug_compression:
                        # 调试模式：直接调用，让调试信息输出到控制台
                        try:
                            answer = qwen_session.generate_stream()
                            
                            # 在控制台打印完整答案
                            print(answer)
                            print("\n")  # 换行
                            
                            # 在Web界面显示答案
                            placeholder.markdown(answer)
                            
                           
                            
                        except Exception as e:
                            error_msg = f"❌ 生成失败: {e}"
                            print(f"\n{error_msg}\n")
                            placeholder.markdown(error_msg)
                            answer = error_msg
                    else:
                        # 非调试模式：流式输出
                        import io
                        from contextlib import redirect_stdout
                        
                        answer = ""
                        old_stdout = sys.stdout
                        
                        # 重置生成字符计数
                        st.session_state.chars_generated = 0
                        
                        try:
                            # 创建一个自定义的输出流来捕获打印内容
                            class StreamCapture:
                                def __init__(self, placeholder, qwen_session, stats_placeholder):
                                    self.placeholder = placeholder
                                    self.qwen_session = qwen_session
                                    self.stats_placeholder = stats_placeholder
                                    self.content = ""
                                    self.update_counter = 0
                                    # 记录生成的字符数（用于估算token增量）
                                    st.session_state.chars_generated = 0
                                
                                def write(self, text):
                                    if text and text != '\n':
                                        self.content += text
                                        self.placeholder.markdown(self.content)
                                        # 累积生成的字符数
                                        st.session_state.chars_generated += len(text)
                                        
                                        # 每12个字符更新一次统计（约等于每3-4个token）
                                        self.update_counter += len(text)
                                        if self.update_counter >= 12:
                                            display_compression_stats(self.stats_placeholder, self.qwen_session)
                                            self.update_counter = 0
                                
                                def flush(self):
                                    pass
                            
                            capture = StreamCapture(placeholder, qwen_session, stats_placeholder)
                            sys.stdout = capture
                            
                            # 调用生成方法（它会打印到我们的捕获流）
                            full_answer = qwen_session.generate_stream()
                            answer = full_answer
                            
                            # 最后再更新一次统计
                            display_compression_stats(stats_placeholder, qwen_session)
                            
                            
                        except Exception as e:
                            error_msg = f"❌ 生成失败: {e}"
                            answer = error_msg
                        finally:
                            sys.stdout = old_stdout
                        
                        # 确保显示完整答案
                        if answer:
                            placeholder.markdown(answer)
                            # 在控制台也打印完整答案
                            print(answer)
                            print("\n")
                        
                        # 清理生成状态标记
                        if 'chars_generated' in st.session_state:
                            del st.session_state.chars_generated
                    
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()



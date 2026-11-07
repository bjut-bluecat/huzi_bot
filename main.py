# -*- coding: utf-8 -*-
"""
户晨风对话机器人 - 主程序
机器人 = 户晨风本人，以第一人称与用户对话
"""
import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BOT_NAME, LOG_LEVEL
from chatbot_engine import HuZiChatbot


def setup_logging(level=LOG_LEVEL):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def print_banner():
    banner = r"""
    ╔══════════════════════════════════════════════════╗
    ║          户子 AI 对话机器人                    ║
    ║                                                  ║
    ║   "我就是个直播的，聊聊天，吹吹牛。"             ║
    ║                                                  ║
    ║   我就是户子，你有什么想问的？直接说。         ║
    ║                                                  ║
    ║   命令:                                          ║
    ║     quit/exit  - 退出                            ║
    ║     stats      - 查看统计                        ║
    ║     search <词> - 搜索语录                       ║
    ║     clear      - 清空对话历史                    ║
    ║     help       - 帮助                            ║
    ╚══════════════════════════════════════════════════╝
    """
    print(banner)


def interactive_mode(chatbot):
    """命令行交互模式 - 和户子聊天"""
    print_banner()
    print("\n户子> 你好你好，来了啊。有什么想问的直接说，别铺垫半天。\n")
    
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n户子> 行，那就这样，拜拜，生活愉快。")
            break
        
        if not user_input:
            continue
        
        cmd = user_input.lower()
        if cmd in ('quit', 'exit', 'q', '退出'):
            print("\n户子> 好，再见再见，下次再来啊。")
            break
        elif cmd == 'stats':
            stats = chatbot.get_stats()
            print("\n--- 户子大脑 ---")
            print("  检索语料:   {} 条".format(stats['corpus_size']))
            print("  问答对:     {} 条".format(stats['qa_pairs']))
            print("  户子回复: {} 条".format(stats['hu_responses']))
            print("  总对话轮次: {} 条".format(stats['total_turns']))
            print("  提取观点:   {} 条".format(stats['opinions_extracted']))
            print("  索引话题:   {} 个".format(stats['topics_indexed']))
            print("  知识条目:   {} 条".format(stats['knowledge_entries']))
            print("  对话历史:   {} 轮".format(stats['history_length']))
            print("--------------\n")
            continue
        elif cmd == 'clear':
            chatbot.clear_history()
            print("\n户子> 好，历史清空了。重新来，说吧。\n")
            continue
        elif cmd == 'help':
            print("\n--- 帮助 ---")
            print("  直接打字    - 和户子聊天")
            print("  search <词> - 搜索户子语录")
            print("  stats       - 查看统计信息")
            print("  clear       - 清空对话历史")
            print("  quit        - 退出")
            print("--------------\n")
            continue
        elif cmd.startswith('search '):
            query = user_input[7:].strip()
            if query:
                results = chatbot.search_quotes(query)
                if results:
                    print("\n--- 户子语录 ---")
                    for i, r in enumerate(results, 1):
                        text = r['text']
                        if len(text) > 150:
                            text = text[:150] + '……'
                        print("  {}. [相关度: {:.3f}] {}".format(i, r['score'], text))
                    print("-----------------\n")
                else:
                    print("\n户子> 没找到相关的，换个词试试。\n")
            continue
        
        # 和户子聊天
        response = chatbot.chat(user_input)
        print("\n户子> {}\n".format(response))


def web_mode(chatbot, host='127.0.0.1', port=8080):
    """Web 界面模式"""
    try:
        from flask import Flask, request, jsonify, render_template_string
    except ImportError:
        print("Web 模式需要 flask: pip install flask")
        print("切换到命令行模式...")
        interactive_mode(chatbot)
        return
    
    app = Flask(__name__)
    
    HTML = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>户子 AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
                background: #0d1117;
                color: #e6edf3;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            .header {
                background: #161b22;
                padding: 12px 20px;
                border-bottom: 1px solid #30363d;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .avatar {
                width: 40px; height: 40px;
                border-radius: 50%;
                background: #e94560;
                display: flex; align-items: center; justify-content: center;
                font-size: 20px; font-weight: bold;
            }
            .header-info h1 { font-size: 16px; }
            .header-info p { font-size: 11px; color: #8b949e; }
            .chat {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .msg {
                max-width: 80%;
                padding: 10px 14px;
                border-radius: 14px;
                line-height: 1.6;
                font-size: 14px;
                word-wrap: break-word;
                white-space: pre-wrap;
            }
            .msg-user {
                align-self: flex-end;
                background: #1f6feb;
                color: #fff;
                border-bottom-right-radius: 4px;
            }
            .msg-bot {
                align-self: flex-start;
                background: #21262d;
                color: #e6edf3;
                border-bottom-left-radius: 4px;
                border: 1px solid #30363d;
            }
            .msg-bot .name { color: #e94560; font-weight: 600; font-size: 12px; margin-bottom: 4px; }
            .input-area {
                background: #161b22;
                padding: 12px 16px;
                display: flex;
                gap: 8px;
                border-top: 1px solid #30363d;
            }
            .input-area input {
                flex: 1;
                padding: 10px 14px;
                border: 1px solid #30363d;
                border-radius: 20px;
                background: #0d1117;
                color: #e6edf3;
                font-size: 14px;
                outline: none;
            }
            .input-area input:focus { border-color: #1f6feb; }
            .input-area button {
                padding: 10px 20px;
                border: none;
                border-radius: 20px;
                background: #e94560;
                color: #fff;
                font-size: 14px;
                cursor: pointer;
            }
            .input-area button:hover { background: #c73e54; }
            .input-area button:disabled { background: #333; cursor: not-allowed; }
            .typing { color: #8b949e; font-style: italic; padding: 6px 14px; font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="avatar">户</div>
            <div class="header-info">
                <h1>户子</h1>
                <p>AI 对话机器人 · 基于直播连麦文字稿训练</p>
            </div>
        </div>
        <div class="chat" id="chat">
            <div class="msg msg-bot">
                <div class="name">户子</div>
                你好你好，来了啊。有什么想问的直接说，别铺垫半天。
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="inp" placeholder="跟户子说点什么..." autocomplete="off">
            <button id="btn" onclick="send()">发送</button>
        </div>
        <script>
            const chat = document.getElementById('chat');
            const inp = document.getElementById('inp');
            const btn = document.getElementById('btn');
            inp.addEventListener('keypress', e => { if (e.key === 'Enter') send(); });
            
            function addMsg(text, isUser) {
                const d = document.createElement('div');
                d.className = 'msg ' + (isUser ? 'msg-user' : 'msg-bot');
                if (!isUser) {
                    const n = document.createElement('div');
                    n.className = 'name';
                    n.textContent = '户子';
                    d.appendChild(n);
                }
                d.appendChild(document.createTextNode(text));
                chat.appendChild(d);
                chat.scrollTop = chat.scrollHeight;
            }
            
            function addTyping() {
                const d = document.createElement('div');
                d.className = 'typing';
                d.id = 'typing';
                d.textContent = '户子正在想...';
                chat.appendChild(d);
                chat.scrollTop = chat.scrollHeight;
            }
            
            async function send() {
                const text = inp.value.trim();
                if (!text) return;
                addMsg(text, true);
                inp.value = '';
                btn.disabled = true;
                addTyping();
                try {
                    const r = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: text})
                    });
                    const data = await r.json();
                    document.getElementById('typing')?.remove();
                    addMsg(data.reply, false);
                } catch(e) {
                    document.getElementById('typing')?.remove();
                    addMsg('网络错误，再试试。', false);
                }
                btn.disabled = false;
                inp.focus();
            }
        </script>
    </body>
    </html>
    '''
    
    @app.route('/')
    def index():
        return render_template_string(HTML)
    
    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        data = request.get_json()
        msg = data.get('message', '')
        reply = chatbot.chat(msg)
        return jsonify({'reply': reply})
    
    @app.route('/api/search')
    def api_search():
        q = request.args.get('q', '')
        results = chatbot.search_quotes(q)
        return jsonify({'results': results})
    
    @app.route('/api/stats')
    def api_stats():
        return jsonify(chatbot.get_stats())
    
    print("\n" + "=" * 50)
    print("  户子 AI 已启动")
    print("  浏览器打开: http://{}:{}".format(host, port))
    print("=" * 50 + "\n")
    
    app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description='户子 AI 对话机器人')
    parser.add_argument('--mode', '-m', choices=['cli', 'web'], default='cli')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--log-level', default='WARNING')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("正在启动户子 AI...")
    
    chatbot = HuZiChatbot(use_cache=not args.no_cache)
    
    stats = chatbot.get_stats()
    logger.info("初始化完成: {} 条语料, {} 条观点, {} 个话题".format(
        stats['corpus_size'], stats['opinions_extracted'], stats['topics_indexed']))
    
    if args.mode == 'web':
        web_mode(chatbot, args.host, args.port)
    else:
        interactive_mode(chatbot)


if __name__ == '__main__':
    main()

# app/ui_gradio.py

from __future__ import annotations
import gradio as gr

from src.rag_pipeline import RAGPipeline


_pipeline = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def chat_fn(query: str, history: str):
    """
    history 這邊先簡單當成一個 string，
    你之後可以改成 list of (user, assistant) 再自己串。
    """
    pipeline = get_pipeline()
    answer = pipeline.answer(query, history=history)
    # 回傳給 gradio 顯示
    new_history = history + f"\nUser: {query}\nAssistant: {answer}\n"
    return answer, new_history


def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Personal Research Notes Assistant")

        state = gr.State(value="")  # 用來存 history 的簡單 string

        with gr.Row():
            with gr.Column(scale=3):
                chatbox = gr.Textbox(
                    label="Question",
                    lines=4,
                    placeholder="請輸入你對論文/筆記的問題...",
                )
                submit_btn = gr.Button("Send")
            with gr.Column(scale=4):
                answer_box = gr.Textbox(
                    label="Answer",
                    lines=12,
                )

        def on_submit(query, history):
            ans, new_history = chat_fn(query, history)
            return ans, new_history

        submit_btn.click(
            fn=on_submit,
            inputs=[chatbox, state],
            outputs=[answer_box, state],
        )

    demo.launch()

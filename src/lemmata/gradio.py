import time
import requests
import typing

import gradio as gr

from lemmata.chain import ChatSession


"""A javascript function to get url parameters for the gradio web server."""
get_window_url_params_js = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);
    return url_params;
    }
"""
no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def http_bot(
    state: ChatSession,
    text,
    image,
    video,
    num_frames,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
) -> typing.Tuple[gr.Button, ...]:
    agent_chain = state.get_agent_chain(
        top_p=float(top_p),
        temperature=float(temperature),
        # max_tokens=min(int(max_output_tokens), 1536)
    )
    history = state.feed(text, agent_chain)

    return (state, history) + (enable_btn,) * 7


@typing.no_type_check
def add_text_http_bot(
    state,
    text,
    image,
    video,
    num_frames,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
):
    if len(text) <= 0 and image is None and video is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5

    if image is not None:
        if "<image>" not in text:
            text = text + "\n<image>"
        text = (text, image)

    if video is not None:
        if "<|video|>" not in text:
            text = text + "\n<|video|>"
        text = (text, video)

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False

    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5
        return

    prompt = state.get_prompt(num_frames)
    prompt = after_process_image(prompt)
    prompt = after_process_video(prompt)
    prompt = prompt.replace("Human: \n", "")

    images = state.get_images()
    videos = state.get_videos(num_frames)

    data = {
        "text_input": prompt,
        "images": images if len(images) > 0 else [],
        "videos": videos if len(videos) > 0 else [],
        "video": video if video is not None else None,
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        },
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    try:
        for chunk in model.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException:
        state.messages[-1][-1] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5


@typing.no_type_check
def regenerate_http_bot(
    state,
    num_frames,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
):
    state.messages[-1][-1] = None
    state.skip_next = False
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    prompt = after_process_image(state.get_prompt(num_frames))
    images = state.get_images()
    videos = state.get_videos(num_frames)

    data = {
        "text_input": prompt,
        "images": images if len(images) > 0 else [],
        "videos": videos if len(videos) > 0 else [],
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        },
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    try:
        for chunk in model.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException:
        state.messages[-1][-1] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5


def upvote_last_response(state: ChatSession, model_selector: str, request: gr.Request) -> typing.Tuple[gr.Button, ...]:
    state.vote_last_response("upvote", request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state: ChatSession, model_selector: str, request: gr.Request) -> typing.Tuple[gr.Button, ...]:
    state.vote_last_response("downvote", request)
    return ("",) + (disable_btn,) * 3


def clear_history(state: ChatSession, request: gr.Request) -> typing.Tuple[gr.Button, ...]:
    state.clear()

    return (state, state.history, "", None, None) + (disable_btn,) * 5


def build_gradio(chat_session: ChatSession, title: str = "Lemmata") -> gr.Blocks:
    # with gr.Blocks(title="mPLUG-Owl🦉", theme=gr.themes.Base(), css=css) as demo:
    with gr.Blocks(title=title) as ui:  # , css=css
        state = gr.State(value=chat_session)

        # gr.Markdown(SHARED_UI_WARNING)
        # gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Image(type="pil")
                videobox = gr.Video()

                with gr.Accordion("Parameters", open=True, visible=False) as parameter_row:
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        interactive=True,
                        label="Top K",
                    )
                    top_p = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                        label="Top p",
                    )
                    length_penalty = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="length_penalty",
                    )
                    num_beams = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        interactive=True,
                        label="Beam Size",
                    )
                    no_repeat_ngram_size = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        interactive=True,
                        label="no_repeat_ngram_size",
                    )
                    num_frames = gr.Slider(
                        minimum=8,
                        maximum=32,
                        value=8,
                        step=4,
                        interactive=True,
                        label="Number of Frames",
                    )
                    do_sample = gr.Checkbox(interactive=True, value=True, label="do_sample")

                # gr.Markdown(tos_markdown)

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(elem_id="chatbot", visible=False).style(height=1000)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            visible=False,
                        ).style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        # gr.Examples(examples=[
        #    [f"examples/monday.jpg", "Explain why this meme is funny."],
        #    [f'examples/rap.jpeg', 'Can you write me a master rap song that rhymes very well based on this image?'],
        #    [f'examples/titanic.jpeg', 'What happened at the end of this movie?'],
        #    [f'examples/vga.jpeg', 'What is funny about this image? Describe it panel by panel.'],
        #    [f'examples/mug_ad.jpeg', 'We design new mugs shown in the image. Can you help us write an advertisement?'],
        #    [f'examples/laundry.jpeg', 'Why this happens and how to fix it?'],
        #    [f'examples/ca.jpeg', "What do you think about the person's behavior?"],
        #    [f'examples/monalisa-fun.jpg', 'Do you know who drew this painting?​'],
        # ], inputs=[imagebox, textbox])

        # gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        gr.HTML("<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>")

        btn_list = [upvote_btn, downvote_btn, regenerate_btn, clear_btn]
        parameter_list = [
            num_frames,
            max_output_tokens,
            temperature,
            top_k,
            top_p,
            num_beams,
            no_repeat_ngram_size,
            length_penalty,
            do_sample,
        ]
        upvote_btn.click(upvote_last_response, [state], [textbox, upvote_btn, downvote_btn])
        downvote_btn.click(downvote_last_response, [state], [textbox, upvote_btn, downvote_btn])
        # regenerate_btn.click(regenerate, state,
        #     [state, chatbot, textbox, imagebox, videobox] + btn_list).then(
        #     http_bot, [state] + parameter_list,
        #     [state, chatbot] + btn_list)
        regenerate_btn.click(
            regenerate_http_bot,
            [state, *parameter_list],
            [state, chatbot, textbox, imagebox, videobox, *btn_list],
        )

        clear_btn.click(clear_history, [state], [state, chatbot, textbox, imagebox, videobox, *btn_list])

        textbox.submit(
            http_bot,  # add_text_http_bot,
            [state, textbox, imagebox, videobox, *parameter_list],
            [state, chatbot, textbox, imagebox, videobox, *btn_list],
        )

        submit_btn.click(
            http_bot,  # add_text_http_bot,
            [state, textbox, imagebox, videobox, *parameter_list],
            [state, chatbot, textbox, imagebox, videobox, *btn_list],
        )

        ui.load(
            load_gradio,
            [url_params],
            [chatbot, textbox, submit_btn, button_row, parameter_row],
            _js=get_window_url_params_js,
        )

    return ui


def load_gradio(url_params: dict, request: gr.Request) -> typing.Tuple[gr.Button, ...]:
    dropdown_update = gr.Dropdown.update(visible=True)

    return (
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )

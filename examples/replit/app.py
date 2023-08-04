from __future__ import annotations

import time
import uuid
import json
import multiprocessing
from functools import partial
from threading import Lock
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Iterator,
    AsyncIterator,
    Sequence,
)
from os import environ

from typing_extensions import TypedDict, Literal

import numpy as np
import numpy.typing as npt

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse

from main import ReplitModel, ReplitSentencepieceTokenizer


## Types
class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChunk(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]


class Completion(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class OpenAIify:
    def __init__(
        self,
        model: ReplitModel,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ):
        self.model = model
        self.cancel_callback = cancel_callback

    def tokenize(self, text: str) -> List[int]:
        return self.model.tokenize(text)

    def detokenize(self, tokens: List[int]) -> str:
        return self.model.detokenize(tokens)

    def generate(
        self,
        tokens: Sequence[int],
        top_p: float = 0.95,
        temperature: float = 0.80,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Iterator[int]:
        return self.model.generate(
            tokens,
            top_p=top_p,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    def _create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        model: Optional[str] = None,
    ) -> Union[Iterator[Completion], Iterator[CompletionChunk]]:
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        completion_tokens: List[int] = []
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens: List[int] = self.tokenize(prompt)
        text: str = ""
        returned_tokens: int = 0
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        model_name: str = model if model is not None else "replit-code-v1-3b"

        # Truncate prompt if it is too long
        max_tokens = min(
            max_tokens, max(0, self.model.max_seq_len - len(prompt_tokens) - 1)
        )
        if len(prompt_tokens) + max_tokens > self.model.max_seq_len:
            raise ValueError(
                f"Requested tokens exceed context window of {self.model.max_seq_len}"
            )

        stop_sequences = stop if stop != [] else []
        finish_reason = "length"
        for token in self.generate(
            prompt_tokens,
            top_p=top_p,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ):
            if token == self.eos_token():
                text = self.detokenize(completion_tokens)
                finish_reason = "stop"
                break

            if self.cancel_callback is not None and self.cancel_callback():
                text = self.detokenize(completion_tokens)
                finish_reason = "stop"
                break

            completion_tokens.append(token)

            all_text = self.detokenize(completion_tokens)
            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = all_text[: all_text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                remaining_tokens = completion_tokens[returned_tokens:]
                remaining_text = self.detokenize(remaining_tokens)
                remaining_length = len(remaining_text)

                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                first_stop_position = 0
                for s in stop_sequences:
                    for i in range(min(len(s), remaining_length), 0, -1):
                        if remaining_text.endswith(s[:i]):
                            if i > first_stop_position:
                                first_stop_position = i
                            break

                token_end_position = 0
                for token in remaining_tokens:
                    token_end_position += len(self.detokenize([token]))
                    # Check if stop sequence is in the token
                    if token_end_position >= (
                        remaining_length - first_stop_position - 1
                    ):
                        break
                    logprobs_or_none: Optional[CompletionLogprobs] = None
                    if logprobs is not None:
                        token_str = self.detokenize([token])
                        text_offset = len(prompt) + len(
                            self.detokenize(completion_tokens[:returned_tokens])
                        )
                        token_offset = len(prompt_tokens) + returned_tokens
                        logits = self.model.scores[token_offset - 1, :].tolist()
                        current_logprobs = self.logits_to_logprobs(logits)
                        sorted_logprobs = list(
                            sorted(
                                zip(current_logprobs, range(len(current_logprobs))),
                                reverse=True,
                            )
                        )
                        top_logprob = {
                            self.detokenize([i]): logprob
                            for logprob, i in sorted_logprobs[:logprobs]
                        }
                        top_logprob[token_str] = current_logprobs[int(token)]
                        logprobs_or_none = {
                            "tokens": [self.detokenize([token])],
                            "text_offset": [text_offset],
                            "token_logprobs": [sorted_logprobs[int(token)][0]],
                            "top_logprobs": [top_logprob],
                        }
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": self.detokenize([token]),
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": None,
                            }
                        ],
                    }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens)
                finish_reason = "length"
                break

        if stream:
            remaining_tokens = completion_tokens[returned_tokens:]
            all_text = self.detokenize(remaining_tokens)
            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                end = min(all_text.index(stop) for stop in any_stop)
            else:
                end = len(all_text)

            token_end_position = 0
            for token in remaining_tokens:
                token_end_position += len(self.detokenize([token]))

                logprobs_or_none: Optional[CompletionLogprobs] = None
                if logprobs is not None:
                    token_str = self.detokenize([token])
                    text_offset = len(prompt) + len(
                        self.detokenize(completion_tokens[:returned_tokens])
                    )
                    token_offset = len(prompt_tokens) + returned_tokens - 1
                    logits = self.model.scores[token_offset, :].tolist()
                    current_logprobs = self.logits_to_logprobs(logits)
                    sorted_logprobs = list(
                        sorted(
                            zip(current_logprobs, range(len(current_logprobs))),
                            reverse=True,
                        )
                    )
                    top_logprob = {
                        self.detokenize([i]): logprob
                        for logprob, i in sorted_logprobs[:logprobs]
                    }
                    top_logprob[token_str] = current_logprobs[int(token)]
                    logprobs_or_none = {
                        "tokens": [self.detokenize([token])],
                        "text_offset": [text_offset],
                        "token_logprobs": [sorted_logprobs[int(token)][0]],
                        "top_logprobs": [top_logprob],
                    }

                if token_end_position >= end:
                    last_text = self.detokenize([token])
                    if token_end_position == end - 1:
                        break
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": last_text[
                                    : len(last_text) - (token_end_position - end)
                                ],
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    break
                returned_tokens += 1
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": self.detokenize([token]),
                            "index": 0,
                            "logprobs": logprobs_or_none,
                            "finish_reason": finish_reason
                            if returned_tokens == len(completion_tokens)
                            else None,
                        }
                    ],
                }
            return

        text_str = text

        if echo:
            text_str = prompt + text_str

        if suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0 if echo else len(prompt)
            token_offset = 0 if echo else len(prompt_tokens[1:])
            text_offsets: List[int] = []
            token_logprobs: List[Optional[float]] = []
            tokens: List[str] = []
            top_logprobs: List[Optional[Dict[str, float]]] = []

            if echo:
                # Remove leading BOS token
                all_tokens = prompt_tokens[1:] + completion_tokens
            else:
                all_tokens = completion_tokens

            all_token_strs = [self.detokenize([token]) for token in all_tokens]
            all_logprobs = [
                self.logits_to_logprobs(row.tolist()) for row in self.model.scores
            ][token_offset:]
            for token, token_str, logprobs_token in zip(
                all_tokens, all_token_strs, all_logprobs
            ):
                text_offsets.append(text_offset)
                text_offset += len(token_str)
                tokens.append(token_str)
                sorted_logprobs = list(
                    sorted(
                        zip(logprobs_token, range(len(logprobs_token))), reverse=True
                    )
                )
                token_logprobs.append(sorted_logprobs[int(token)][0])
                top_logprob: Optional[Dict[str, float]] = {
                    self.detokenize([i]): logprob
                    for logprob, i in sorted_logprobs[:logprobs]
                }
                top_logprob.update({token_str: logprobs_token[int(token)]})
                top_logprobs.append(top_logprob)
            # Weird idosincracy of the OpenAI API where
            # token_logprobs and top_logprobs are null for
            # the first token.
            if echo and len(all_tokens) > 0:
                token_logprobs[0] = None
                top_logprobs[0] = None
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }

    def create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        model: Optional[str] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_or_chunks = self._create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream,
            model=model,
        )
        if stream:
            chunks: Iterator[CompletionChunk] = completion_or_chunks
            return chunks
        completion: Completion = next(completion_or_chunks)  # type: ignore
        return completion

    def eos_token(self):
        return self.model.eos_token()

    def logits_to_logprobs(
        self, logits: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return np.exp(logits) / (np.sum(np.exp(logits)))  # type: ignore


class Settings(BaseSettings):
    model_file: str
    n_gpu_layers: int = 32
    n_batch: int = 2048
    n_threads: int = max(multiprocessing.cpu_count() // 2, 1)
    sentencepiece_model: Optional[str] = None


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: int = Field(
        default=16,
        ge=1,
        le=2048,
        description="The maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Adjust the randomness of the generated text.\n\n"
        + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
        + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
    )
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the results as they are generated. Useful for chatbots.",
    )
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
    )
    logprobs: Optional[int] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = Field(
        description="The model to use for generating completions."
    )
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "prompt": "def fib(n):",
                "stop": ["\n\n"],
                "temperature": 0,
                "max_tokens": 34,
            }
        }


settings = Settings(model_file=environ.get("MODEL"))  # type: ignore
app = FastAPI(
    title="Code Completion API",
    description="""
## Editor Setup

### VSCode

Add the following to your `settings.json`:

```json
{
    "github.copilot.advanced": {
        "debug.testOverrideProxyUrl": "http://localhost:8000",
        "debug.overrideProxyUrl": "http://localhost:8000"
    }
}
```

### Vim / Neovim

Add the following to your vimrc or init.vim:

```
let g:copilot_proxy = 'localhost:8000'
let g:copilot_strict_ssl = 0
```
""",
)
outer_lock = Lock()
inner_lock = Lock()

tokenizer = (
    ReplitSentencepieceTokenizer(settings.sentencepiece_model)
    if settings.sentencepiece_model
    else None
)


def cancel_callback():
    return outer_lock.locked()


model = OpenAIify(
    ReplitModel.init_from_file(
        model_file=settings.model_file,
        n_gpu_layers=settings.n_gpu_layers,
        tokenizer=tokenizer,
        cancel_callback=cancel_callback,
    ),
    # check if any other requests are pending in the same thread and cancel the stream if so
    cancel_callback=cancel_callback,
)


def get_model():
    # NOTE: This double lock allows the currently streaming model to check
    # if any other requests are pending in the same thread and cancel the
    # stream if so.
    outer_lock.acquire()
    release_outer_lock = True
    try:
        inner_lock.acquire()
        try:
            outer_lock.release()
            release_outer_lock = False
            yield model
        finally:
            inner_lock.release()
    finally:
        if release_outer_lock:
            outer_lock.release()


# Used to support copilot.vim
@app.get("/copilot_internal/v2/token")
def get_copilot_token():
    content = {"token": "1", "expires_at": 2600000000, "refresh_in": 900}
    return dict(status_code=200, content=content)


CreateCompletionResponse = create_model_from_typeddict(Completion)


# Used to support copilot.vim
@app.post(
    "/v1/engines/copilot-codex/completions",
    # response_model=CreateCompletionResponse,
)
@app.post(
    "/v1/completions",
    # response_model=CreateCompletionResponse,
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    model: ReplitModel = Depends(get_model),
):
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    exclude = {
        "n",
        "best_of",
        "logit_bias",
        "user",
    }
    kwargs = body.dict(exclude=exclude)
    if body.stream:
        send_chan, recv_chan = anyio.create_memory_object_stream(10)

        async def event_publisher(
            inner_send_chan: MemoryObjectSendStream[Dict[str, Union[str, bool]]]
        ):
            async with inner_send_chan:
                try:
                    iterator: Iterator[CompletionChunk] = await run_in_threadpool(model.create_completion, **kwargs)  # type: ignore
                    async_iterator: AsyncIterator[
                        CompletionChunk
                    ] = iterate_in_threadpool(iterator)
                    async for chunk in async_iterator:
                        await inner_send_chan.send(dict(data=json.dumps(chunk)))
                        if await request.is_disconnected():
                            raise anyio.get_cancelled_exc_class()()
                    await inner_send_chan.send(dict(data="[DONE]"))
                except anyio.get_cancelled_exc_class() as e:
                    print("disconnected")
                    with anyio.move_on_after(1, shield=True):
                        print(
                            f"Disconnected from client (via refresh/close) {request.client}"
                        )
                        await inner_send_chan.send(dict(closing=True))
                        raise e

        return EventSourceResponse(
            recv_chan, data_sender_callable=partial(event_publisher, send_chan)
        )
    else:
        completion: Completion = await run_in_threadpool(model.create_completion, **kwargs)  # type: ignore
        return completion

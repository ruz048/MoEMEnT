# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict
import yaml
import time
import re
import openai
from openai import AzureOpenAI
from accelerate.logging import get_logger
from openai import (
    RateLimitError,
    #InvalidRequestError,
    Timeout,
    APIConnectionError,
    APIError,
    #ServiceUnavailableError,
)

from .base import BaseEval

logger = get_logger(__name__)


def get_retry_time(err_info):
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 5


class ModelEval(BaseEval):
    """Compute evaluate metrics with GPT4"""

    def __init__(self, config: str | dict, judge_fn: Callable, format_fn: Callable):
        super().__init__()

        self.config = self.load_config(config)

        self.judge_fn = judge_fn
        self.format_fn = format_fn

        #self.model = 'gpt-4o'
        self.model = 'gpt-4-turbo-v'
        self.client = AzureOpenAI(
    api_key="YOUR API KEY",  
    api_version="2024-05-01-preview",
    azure_endpoint = "YOUR ENDPOINT"
    )

    def load_config(self, config: str | dict) -> Dict:
        if isinstance(config, dict):
            return config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    def chat_completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        success = False
        cnt = 0
        while not success:
            time.sleep(0.5)
            try:
                response = self.client.chat.completions.create(
                    model = self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                #print(response)
                success = True
            except RateLimitError as e:
                #logger.debug(e, exc_info=True)
                print(e)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except Timeout as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except:
                #logger.warning(e, exc_info=True)
                print('Error:{}',format(str(cnt)))
                if cnt>10:
                    exit()
                cnt+=1
                time.sleep(60)
                response = {"choices": []}

        try:
            rslts = [i.message.content for i in response.choices]
        except Exception as e:
            logger.warning(e, exc_info=True)
            rslts = []
        print(success)
        return rslts

    def completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"],
    ):
        success = False
        
        while not success:
            time.sleep(0.1)
            try:
                response = self.client.completions.creat(
                    model=self.model,
                    prompt=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                success = True
            except RateLimitError as e:
                logger.debug(e, exc_info=True)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except Timeout as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"choices": []}

        rslts = [i.text for i in response.choices]
        return rslts

    def _compute_score(self, prediction: str = None, **kwargs):
        messages = self.format_fn(prediction, chat=self.config["chat"])

        if self.config["chat"]:
            response = self.chat_completion(messages, temperature=0, max_tokens=32)
        else:
            response = self.completion(messages, temperature=0, max_tokens=32)

        if len(response) > 0:
            asr = self.judge_fn(response[0])
        else:
            asr = -1
        return asr

    def _batch_compute_score(self, predictions: str = None, **kwargs):
        messages = [
            self.format_fn(prediction, chat=self.config["chat"])
            for prediction in predictions
        ]

        responses = self.completion(messages, temperature=0, max_tokens=32)

        if len(responses) > 0:
            asrs = [self.judge_fn(response) for response in responses]
        else:
            asrs = [-1] * len(messages)
        return asrs

    def add_batch(self, *, predictions=None, **kwargs):
        if self.config["chat"]:
            batch_asrs = []
            for pred in predictions:
                asr = self._compute_score(prediction=pred)
                batch_asrs.append(asr)

            self.asrs.extend(batch_asrs)
        else:
            batch_asrs = self._batch_compute_score(predictions=predictions)
            self.asrs.extend(batch_asrs)

        return batch_asrs

import asyncio

from openai import AsyncOpenAI

from typing import Any, Dict, List, Optional

from vllm import SamplingParams,LLM

class AsyncLLMServer:
    '''异步 LLM 服务客户端，基于 OpenAI 兼容 API 进行对话与批量推理。'''

    def __init__(self, model_name: str, api_key: str="em", base_url: Optional[str] = None, concurrency: int=1,**kwargs):
        '''初始化异步 LLM 服务客户端，连接远程 API 并获取模型最大长度等配置。

        Descriptions:
            创建 AsyncOpenAI 客户端，请求 base_url 获取模型 max_model_len，
            并设置并发数、max_tokens、temperature、是否启用 thinking 等参数。

        Args:
            model_name: 模型名称。
            api_key: API 密钥，默认 "em"。
            base_url: API 基础 URL，如 http://localhost:7899/v1。
            concurrency: 并发请求数，默认 1，内部会用于信号量限制。
            **kwargs: 可选参数，如 max_tokens、temperature、enable_thinking、tokenizer。

        Returns:
            None.
        '''
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        import requests
        url = f"{base_url}/models/"
        response = requests.get(url)
        #FIXME:暂时假设只部署一个模型
        self.max_model_length = response.json()["data"][0]["max_model_len"]
        self.model_name = model_name
        self.concurrency = concurrency if concurrency is not None else 8 #默认8线程
        self.sem = None
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.0)
        self.enable_thinking = kwargs.get("enable_thinking", False)
        self.tokenizer = kwargs.get("tokenizer", None)
    def set_sem(self):
        '''设置并发控制的信号量。

        Descriptions:
            根据初始化时的 concurrency 创建 asyncio.Semaphore，用于限制同时进行的请求数。

        Args:
            None.

        Returns:
            None.
        '''
        self.sem = asyncio.Semaphore(self.concurrency)
    async def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,enable_thinking: Optional[bool] = None):
        '''单次对话补全（Chat Completions API）。

        Descriptions:
            调用 OpenAI 兼容的 chat.completions 接口，根据 messages 获取模型回复文本。

        Args:
            messages: 对话消息列表，每项为 role/content 等字段的字典。
            tools: 可选，工具定义列表，当前实现未在请求中传入。
            enable_thinking: 可选，是否启用思考模式，当前实现未使用。

        Returns:
            str: 模型回复内容，即 response.choices[0].message.content。
        '''
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content
    async def _chat(self,prompt: str,max_tokens: int=4096,temperature: float=0.0):
        '''内部单次补全：优先 responses API，失败时回退到 completions API。

        Descriptions:
            先尝试 client.responses.create（input=prompt），失败则使用
            client.completions.create（prompt/max_tokens/temperature）获取文本。

        Args:
            prompt: 已拼接好的完整提示文本。
            max_tokens: 最大生成 token 数，默认 4096。
            temperature: 采样温度，默认 0.0。

        Returns:
            str: 模型生成的文本。任一 API 成功即返回对应 content/text。
        '''
        try:
            response = await self.client.responses.create(
                model=self.model_name,
                input=prompt
            )
            return response.output[0].content[0].text
        except Exception as e:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text
        except Exception as e:
            raise e
    
    async def chat_batch(
        self,
        list_of_messages: List[List[Dict[str, Any]]],
        tools_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
        **kwargs: Any
    ) -> List[str]:
        '''批量对话请求，在并发限制下并行调用 chat_one_async。

        Descriptions:
            为每条消息（及对应 tools）创建异步任务，通过 asyncio.gather 并行执行，
            若已调用 set_sem，则并发数由 sem 限制。

        Args:
            list_of_messages: 多条对话，每条为 List[Dict[str, Any]]。
            tools_list: 可选，与 list_of_messages 等长的工具列表，每项可为 None 或工具定义列表。
            **kwargs: 其他可选参数，会传给 chat_one_async。

        Returns:
            List[str]: 与 list_of_messages 一一对应的模型回复字符串列表。
        '''
        if tools_list is None:
            tools_list = [None] * len(list_of_messages)
        if len(tools_list) != len(list_of_messages):
            raise ValueError("tools_list length must match list_of_messages length")
        # concurrency = getattr(args, "concurrency", 1)
        # sem = asyncio.Semaphore(concurrency)
        # prompts = []
        # for messages,tool in zip(list_of_messages,tools_list):
        #     prompt = args.tokenizer.apply_chat_template(messages, tools=tool, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
        #     prompts.append(prompt)
        # async def run_one(prompt,args):
        #     async with sem:
        #         # prompt_tokens = len(args.tokenizer.encode(prompt))
        #         # rest_tokens = self.max_model_length - prompt_tokens
        #         # if rest_tokens < 0:#会存在feedback的情况，所以会导致rest_tokens小于0
        #         #     return "OverLengthError: The prompt is too long, please shorten it."
        #         # return await self._chat(
        #         #     prompt=prompt,
        #         #     max_tokens=min(args.max_tokens, max(rest_tokens, 0)),
        #         #     temperature=args.temperature
        #         # )

        # tasks = [run_one(p,args) for p in prompts]
        
        tasks = [self.chat_one_async(messages, tools) for messages, tools in zip(list_of_messages, tools_list)]
        return await asyncio.gather(*tasks)

    async def chat_one_async(self, messages, tools,max_tokens: int=None,temperature: float=None,**kwargs: Any):
        '''单条对话的异步执行：模板拼接、长度检查后调用 _chat。

        Descriptions:
            使用 tokenizer.apply_chat_template 将 messages 与 tools 拼成 prompt，
            计算 token 数并与 max_model_length 比较；若超长则返回空字符串，
            否则在可选 sem 控制下调用 _chat 并返回生成文本。

        Args:
            messages: 单条对话消息列表，List[Dict[str, Any]]。
            tools: 可选工具定义列表，Optional[List[Dict[str, Any]]]。
            max_tokens: 最大生成 token 数，None 时使用实例默认值。
            temperature: 采样温度，None 时使用实例默认值。
            **kwargs: 其他可选参数，当前未使用。

        Returns:
            str: 模型回复内容；若 prompt 超长则返回空字符串 ""。
        '''
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        rest_tokens = self.max_model_length - prompt_tokens
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        if rest_tokens < 0:#会存在feedback的情况，所以会导致rest_tokens小于0
            #OverLengthError: The prompt is too long, please shorten it.
            return ""
        if self.sem is not None:
            async with self.sem:
                return await self._chat(
                    prompt=prompt,
                    max_tokens=min(max_tokens, max(rest_tokens, 0)),
                    temperature=temperature
                )
        else:
            return await self._chat(
                prompt=prompt,
                max_tokens=min(max_tokens, max(rest_tokens, 0)),
                temperature=temperature
            )
    
    
# 本地 vLLM 引擎 batch 推理
class VLLMEngine:
    '''本地 vLLM 批量推理引擎，用于单机多请求批量生成。'''

    def __init__(self, model_path, max_tokens=4096, temperature=0.0):
        '''初始化本地 vLLM 引擎与默认采样参数。

        Descriptions:
            加载 model_path 对应的 vLLM 模型，创建固定 max_tokens、temperature 的
            SamplingParams，并获取 tokenizer 供后续模板拼接使用。

        Args:
            model_path: 模型路径或名称，传给 LLM(model=...)。
            max_tokens: 单次生成最大 token 数，默认 4096。
            temperature: 采样温度，默认 0.0。

        Returns:
            None.
        '''
        self.llm = LLM(model=model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.tokenizer = self.llm.get_tokenizer()
        
    def chat_batch(self, list_of_messages, tools, args):
        '''使用本地 vLLM 对多条对话进行批量生成。

        Descriptions:
            对 list_of_messages 与 tools 逐条应用 tokenizer.apply_chat_template
            （使用 args.enable_thinking），得到 prompts 后一次性调用 llm.generate，
            返回每条对话对应的生成文本列表。

        Args:
            list_of_messages: 多条对话，每条为消息列表。
            tools: 与 list_of_messages 等长的工具列表，每项可为 None 或工具定义。
            args: 包含 enable_thinking 等参数的对象。

        Returns:
            List[str]: 每条对话的生成文本，与 list_of_messages 一一对应。
        '''
        prompts = []
        for messages,tool in zip(list_of_messages,tools):
            prompt = self.tokenizer.apply_chat_template(messages, tools=tool, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
            prompts.append(prompt)
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = [out.outputs[0].text for out in outputs]
        return results

if __name__ == "__main__":
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:7899/v1", api_key="local")
    client.chat.completions.create(
                    model="MatchTIR",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                )
    response = client.completions.create(
        model="MatchTIR",
        max_tokens=4096,
        prompt="""<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "geo_relationship_finder", "description": "An advanced tool for discovering and analyzing relationships between geographical locations and various types of landmarks and entities. It offers comprehensive insights into connections between specified locations and nearby entities, with enhanced filtering and output options.", "parameters": {"type": "object", "properties": {"location_name": {"type": "string", "description": "The name of the primary location to find connections for."}, "entity_types": {"type": "array", "items": {"type": "string", "enum": ["park", "zoo", "garden", "landmark", "museum", "historical_site"], "description": "The types of entities to find connections with. Can specify multiple types."}, "description": "The types of entities to find connections with. Defaults to [\'park\'] if not specified."}, "radius": {"type": "number", "description": "The radius in kilometers within which to search for connected entities. Defaults to 5 km."}, "filter_criteria": {"type": "object", "properties": {"popularity": {"type": "string", "enum": ["high", "medium", "low"], "description": "Filter results based on the popularity of the entities."}, "historical_significance": {"type": "boolean", "description": "Filter results to include only historically significant entities."}}, "description": "Optional criteria to filter the search results."}, "output_format": {"type": "string", "enum": ["list", "detailed", "geojson"], "description": "The format of the output. \'list\' provides a simple list of connected entities, \'detailed\' provides more information about each connection, and \'geojson\' provides geospatial data format."}}, "required": ["location_name"]}}}\n{"type": "function", "function": {"name": "historical_figure_identifier", "description": "An advanced tool designed to identify historical figures associated with specific events or constructions. It searches through historical databases and records to provide accurate and detailed information about the individuals involved, with enhanced search precision and customization options.", "parameters": {"type": "object", "properties": {"event_name": {"type": "string", "description": "The name of the event or construction project, e.g., \'Stanley Park\'."}, "time_period": {"type": "string", "description": "The time period during which the event took place, to narrow down the search."}, "location": {"type": "string", "description": "The geographical location of the event, to provide context and improve search accuracy."}, "name_variations": {"type": "boolean", "description": "Whether to consider variations and alternate spellings of names, default is true."}, "output_format": {"type": "string", "enum": ["text", "json"], "description": "The format of the output. Defaults to text (a descriptive answer)."}, "figure_type": {"type": "string", "enum": ["political", "cultural", "scientific", "military", "other"], "description": "The type of historical figure to identify, to refine the search."}, "detail_level": {"type": "string", "enum": ["summary", "detailed"], "description": "The level of detail in the output. Defaults to summary."}, "significance_filter": {"type": "boolean", "description": "Whether to filter results by the significance of the figure\'s role in the event, default is false."}, "information_source": {"type": "string", "description": "The source of information to use for the search, e.g., \'academic\', \'public records\'."}}, "required": ["event_name"]}}}\n{"type": "function", "function": {"name": "extract_first_name", "description": "An advanced tool designed to extract the first name from a full name string. It supports various name formats, including those with middle names and initials, and offers extensive output customization options. The tool also includes enhanced error handling and validation features.", "parameters": {"type": "object", "properties": {"full_name": {"type": "string", "description": "The full name from which the first name will be extracted. Supports names with middle names and initials."}, "output_format": {"type": "string", "enum": ["string", "uppercase", "lowercase", "titlecase"], "description": "The format of the output. Defaults to string (the first name as is)."}, "handle_special_characters": {"type": "boolean", "description": "Indicates whether to handle special characters in the name. Defaults to false."}, "error_handling": {"type": "string", "enum": ["ignore", "warn", "strict"], "description": "The level of error handling for invalid input. Defaults to \'warn\'."}}, "required": ["full_name"]}}}\n{"type": "function", "function": {"name": "count_letters", "description": "An advanced tool for counting the number of letters in a given string. It provides enhanced customization options to ignore specific characters, positions, or even entire words, and offers multiple output formats for diverse use cases.", "parameters": {"type": "object", "properties": {"input": {"type": "string", "description": "The string whose letters are to be counted."}, "ignore_characters": {"type": "array", "items": {"type": "string", "description": "Characters to be ignored in the count, default is none."}}, "ignore_position": {"type": "array", "items": {"type": "string", "enum": ["first", "last", "even", "odd"], "description": "Positions to be ignored in the count, default is none. Can now ignore even or odd positions."}}, "ignore_words": {"type": "array", "items": {"type": "string", "description": "Words to be ignored in the count, default is none."}}, "output_format": {"type": "string", "enum": ["integer", "text", "json"], "description": "The format of the output, either as an integer, text, or JSON. Defaults to integer."}}, "required": ["input"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nYou will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: How many letters (exclude the first and last) are there in the first name of the person who designed and built the park which links Salisbury Woodland Gardens with a zoo?<|im_end|>\n<|im_start|>assistant\n"""
    )
    print(response.choices[0].text)
    pass
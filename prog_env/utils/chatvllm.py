from prog_env.utils.llm_server import AsyncLLMServer, VLLMEngine

class ChatVLLM:
    '''统一的对话接口：根据配置选择本地 VLLMEngine 或远程 AsyncLLMServer 进行批量/单条推理。'''

    def __init__(self, args):
        '''根据 args.engine 初始化本地或远程推理引擎。

        Descriptions:
            engine 为 "local" 时使用 VLLMEngine（model_path/temperature/max_tokens）；
            engine 为 "remote" 时使用 AsyncLLMServer（base_url、concurrency、tokenizer 等）。
            将 args 保存到 self.args 供后续 batch 调用使用。

        Args:
            args: 配置对象，需包含 engine、model_path、base_url、concurrency、
                  max_tokens、temperature、enable_thinking、tokenizer 等属性。

        Returns:
            None.
        '''
        if args.engine == "local":
            self.engine = VLLMEngine(
                model_path=args.model_path,
                temperature=args.temperature,
                max_tokens=args.max_tokens
                )
        elif args.engine == "remote":
            self.engine = AsyncLLMServer(
                model_name="MatchTIR",
                base_url=args.base_url,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                tokenizer=args.tokenizer
            )
        self.args = args

    def chat_open_batch(self, batch_messages, tools):
        '''对多组对话进行批量推理，自动根据引擎类型调用同步或异步接口。

        Descriptions:
            若引擎为 VLLMEngine，直接调用 engine.chat_batch(batch_messages, tools, self.args)；
            若为 AsyncLLMServer，则 asyncio.run(engine.chat_batch(...)) 执行异步批量请求。
            返回与 batch_messages 一一对应的模型回复列表。

        Args:
            batch_messages: 多组对话，每组为 List[Dict[str, Any]] 形式的消息列表。
            tools: 与 batch_messages 等长的工具列表，每项可为 None 或工具定义列表。

        Returns:
            List[str]: 每组对话的模型回复文本列表。
        '''
        if isinstance(self.engine, VLLMEngine):
            raw_outputs = self.engine.chat_batch(batch_messages, tools, self.args)
        elif isinstance(self.engine, AsyncLLMServer):
            import asyncio
            raw_outputs = asyncio.run(self.engine.chat_batch(batch_messages, tools, args=self.args))
        return raw_outputs
    
    def chat_one_sync(self, messages, tools):
        '''单条对话的同步调用，内部转为单元素 batch 后取第一个结果。

        Descriptions:
            将单组 messages 与 tools 包装成 [messages] 与 [tools]，调用 chat_open_batch，
            返回第一个（也是唯一一个）回复字符串。适用于本地或远程引擎的统一同步接口。

        Args:
            messages: 单条对话消息列表，List[Dict[str, Any]]。
            tools: 可选工具定义列表，Optional[List[Dict[str, Any]]]。

        Returns:
            str: 模型对该条对话的回复内容。
        '''
        return self.chat_open_batch([messages], [tools])[0]
    
    async def chat_one_async(self, messages, tools):
        '''单条对话的异步调用，仅支持 AsyncLLMServer 引擎。

        Descriptions:
            委托给 self.engine.chat_one_async(messages, tools)。若 prompt 超长则返回空字符串，
            否则返回模型生成内容。调用前会断言引擎类型为 AsyncLLMServer。

        Args:
            messages: 单条对话消息列表，List[Dict[str, Any]]。
            tools: 可选工具定义列表，Optional[List[Dict[str, Any]]]。

        Returns:
            str: 模型回复内容；若超长则返回空字符串 ""。
        '''
        assert isinstance(self.engine, AsyncLLMServer)
        return await self.engine.chat_one_async(messages, tools)
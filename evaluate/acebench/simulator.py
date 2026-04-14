import asyncio
from copy import deepcopy

from openai import AsyncOpenAI

from prog_env.utils.chatvllm import ChatVLLM
from prog_env.utils.llm_server import AsyncLLMServer

from evaluate.acebench.execution import execute_agent_func_call, snapshot_instances
from evaluate.acebench.metrics import compute_end_to_end_accuracy, compute_process_accuracy
from evaluate.acebench.parser import extract_outermost_bracket_content, looks_like_function_call


MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH = """你是一个AI系统，你的角色为system，请根据给定的API说明和对话历史1..t，为角色system生成在步骤t+1中生成相应的内容。
1 如果上一步提供的信息完整，能够正常进行api的调用，你应该调用的API请求，API请求以[ApiName(key1='value1', key2='value2', ...)]的格式输出,不要在输出中输出任何其他解释或提示或API调用的结果。
如果API参数描述中没有特殊说明，则该参数为非必选参数（用户输入中提及的参数需要包含在输出中，如果未提及，则不需要包含在输出中）。\n如果API参数描述未指定取值格式要求，则该参数取值使用用户原文。
2 如果你得到的信息不完整，需要向user提问，以获得完整的信息。你不能扮演user去回答一些文职的问题，应该及时像user询问。

请注意，如果需要进行api调用，请严格遵守调用规则[ApiName(key1='value1', key2='value2', ...)]，此时不得输出其他文本内容。

角色说明：
user: 用户
agent: 进行API请求调用的AI系统角色
execution: 执行api调用并返回结果
"""

MULTI_TURN_AGENT_PROMPT_USER_ZH = """下面是你可使用的api列表:\n {functions}\n\n对话历史1..t:\n{history}"""

MULTI_TURN_AGENT_PROMPT_SYSTEM_EN = """You are an AI system with the role name "system." Based on the provided API specifications and conversation history from steps 1 to t, generate the appropriate content for step t+1 for the "system" role.
1. If the information provided in the previous step is complete and the API call can be executed normally, you should generate the API request. The API request should be output in the format [ApiName(key1='value1', key2='value2', ...)]. Do not include any other explanations, prompts, or API call results in the output.
   - If the API parameter description does not specify otherwise, the parameter is optional (parameters mentioned in the user input need to be included in the output; if not mentioned, they do not need to be included).
   - If the API parameter description does not specify the required format for the value, use the user's original text for the parameter value.
2. If the information you received is incomplete, you need to ask the user for more information to obtain the complete details. You should not pretend to be the user to answer some clerical questions; instead, promptly ask the user for clarification.

Please note that if an API call is required, strictly adhere to the call format rules [ApiName(key1='value1', key2='value2', ...)] and do not output any other text content.

Role Descriptions:
user: User
agent: The AI system role that makes API requests
execution: Executes the API call and returns results
"""

MULTI_TURN_AGENT_PROMPT_USER_EN = """Below is the list of APIs you can use:\n {functions}\n\nConversation history 1..t:\n{history}"""

MULTI_STEP_AGENT_PROMPT_SYSTEM_ZH = """你是一个AI系统，你的角色为system，请根据给定的API说明和对话历史1..t，为角色system生成在步骤t+1中生成相应的内容。
1 如果上一步提供的信息完整，能够正常进行api的调用，你应该调用的API请求，API请求以[ApiName(key1='value1', key2='value2', ...)]的格式输出，将ApiName替换为实际的API名称，将key1、key2等替换为实际的参数名称，将value1、value2替换为实际参数取值。输出应以方括号"["开头，以方括号"]"结尾。API请求有多个时以英文逗号隔开，比如[ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...), ...]。不要在输出中输出任何其他解释或提示或API调用的结果。\n
如果API参数描述中没有特殊说明，则该参数为非必选参数（用户输入中提及的参数需要包含在输出中，如果未提及，则不需要包含在输出中）。\n如果API参数描述未指定取值格式要求，则该参数取值使用用户原文。
2 当一个任务需要多个步骤才能完成(步骤之间有严格的前后关系)，你需要一步步执行，并根据每一轮execution返回的结果决定下一步如何执行。
3 一般不使用并行调用的方法，也就是一次只调用一个函数。

请注意，如果需要进行api调用，请严格遵守调用规则[ApiName(key1='value1', key2='value2', ...)]，此时不得输出其他内容。
当你认为任务已经完成，请返回"finish conversation"以结束对话。

角色说明：
user: 用户
agent: 进行API请求调用的AI系统角色
execution: 执行api调用并返回结果
"""

MULTI_STEP_AGENT_PROMPT_USER_ZH = """以下是你可以调用的API列表（JSON格式）：{functions}。对话历史：{history}\n"""

MULTI_STEP_AGENT_PROMPT_SYSTEM_EN = """You are an AI system with the role of 'system'. Based on the provided API documentation and the conversation history from steps 1 to t, generate the corresponding content for the 'system' role in step t+1.
1. If the information provided in the previous step is complete and allows for a successful API call, you should output the API request(s) to be called in the format [ApiName(key1='value1', key2='value2', ...)]. Replace ApiName with the actual API name, key1, key2, etc., with the actual parameter names, and value1, value2, etc., with the actual parameter values. The output should start with a square bracket "[" and end with a square bracket "]". If there are multiple API requests, separate them with commas, for example, [ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...), ...]. Do not include any additional explanations, prompts, or API call results in the output.
   - If the API parameter description does not specify otherwise, the parameter is optional (only include parameters mentioned in the user input; if not mentioned, do not include them).
   - If the API parameter description does not specify a required value format, use the user's original input for the parameter value.
2. If a task requires multiple steps to complete (with strict sequential relationships between steps), execute them step by step, and decide how to proceed based on the results returned from each execution.
3. Generally do not use parallel calls, meaning only one function is called at a time.

Please note that if an API call is needed, strictly adhere to the calling rules [ApiName(key1='value1', key2='value2', ...)] and do not output any other content.
When you believe the task is completed, return "finish conversation" to end the dialogue.

Role Descriptions:
user: The user
agent: The AI system role that performs API requests
execution: Executes API calls and returns results
"""

MULTI_STEP_AGENT_PROMPT_USER_EN = """Below is the list of APIs you can call (in JSON format): {functions}. Conversation history: {history}\n"""

SYSTEM_PROMPT_TRAVEL_ZH = """您是一名与agent互动的用户。

Instruction: {instruction}

规则：
- 每次只生成一行内容，以模拟用户的消息。
- 不要一次性透露所有说明内容。只提供当前步骤所需的信息。
- 不要臆测说明中未提供的信息。例如，如果agent询问订单ID，但说明中没有提到，请不要编造订单ID，而是直接表示不记得或没有。
- 当遇到需要信息确认的时候，根据Instruction 中的内容决定是否确认。
- 不要在对话中重复说明内容，而是使用您自己的话来表达相同的信息。
- 尽量使对话自然，保持说明中描述的用户个性。
- 如果说明目标已达成，生成单独一行的 'finish conversation' 消息以结束对话。
- 如果Instruction中要求预定往返航班，则需要在最开始说明意图"预定往返航班"。
"""

SYSTEM_PROMPT_BASE_ZH = """您是一名与agent互动的用户。

Instruction: {instruction}

规则：
- 每次只生成一行内容，以模拟用户的消息。
- 不要一次性透露所有说明内容。只提供当前步骤所需的信息。
- 需要将当前步骤所需的信息提供完整。例如，添加提醒时需要提供提醒的描述，标题和时间等。
- 不要臆测说明中未提供的信息。例如，Instruction中并没有直接指明外卖内容，而随意编造外卖内容。
- 当被询问是否还需要帮助时，一定要确保Instruction中的主要任务是否都已被完成，如果没有，则继续向agent提出下一步任务。
- Instructiuon中出现的名字，即默认用户全名。
- 当agent询问需要删除哪一条短信时，需要按照Instruction中的要求删除短信。
- 你不能主动向agent提供帮助，按 Instruction中的要求回复agent问题，不能编造任何你未知的信息。
- 如果所有任务已完成，生成单独一行的 'finish conversation' 消息以结束对话。
"""

SYSTEM_PROMPT_TRAVEL_EN = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Generate only one line of content each time to simulate the user's message.
- Do not reveal all instruction content at once. Only provide information needed for the current step.
- Do not speculate information not provided in the instructions. For example, if the agent asks for an order ID but it is not mentioned in the instructions, do not fabricate an order ID; instead, directly state that you do not remember or do not have it.
- When information confirmation is needed, decide whether to confirm based on the content in the Instruction.
- Do not repeat instruction content in the conversation; instead, express the same information in your own words.
- Keep the dialogue natural and maintain the user's personality as described in the instructions.
- If the goal in the instructions has been achieved, generate a separate line with the message 'finish conversation' to end the dialogue.
- If the Instruction requires booking a round-trip flight, you need to state the intention "Book a round-trip flight" at the very beginning.
"""

SYSTEM_PROMPT_BASE_EN = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Generate only one line of content each time to simulate the user's message.
- Do not reveal all instruction content at once. Only provide information needed for the current step.
- Ensure that all information needed for the current step is provided completely. For example, when adding a reminder, you need to provide the reminder's description, title, and time, etc.
- Do not speculate information not provided in the Instruction.
- When asked if you need further assistance, make sure whether all main tasks in the Instruction have been completed. If not, continue to provide the next step task to the agent.
- Names appearing in the Instruction are assumed to be the user's full names.
- When the agent asks which message to delete, follow the Instruction's requirements to delete the message.
- You cannot proactively offer help to the agent. Respond to the agent's questions as per the Instruction's requirements, and do not fabricate any information you do not know.
- If all tasks are completed, generate a separate line with the message 'finish conversation' to end the dialogue.
"""


def _remove_prefix(text: str) -> str:
    if text.startswith("user:"):
        return text[5:]
    if text.startswith("agent:"):
        return text[6:]
    return text


def build_history_text(history: list[dict], scenario: str) -> str:
    lines = []
    for item in history:
        sender = item["sender"]
        if sender == "user":
            lines.append(f"user:{item['message']}")
        elif sender == "agent":
            lines.append(f"agent:{item['message']}")
        elif sender == "execution":
            prefix = "execution result:" if scenario == "agent_multi_step" else "execution:"
            lines.append(f"{prefix}{item['message']}")
    return "\n".join(lines) + ("\n" if lines else "")


class AsyncChatResponder:
    def __init__(
        self,
        backend: str,
        model_name: str,
        chat_engine: ChatVLLM | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.chat_engine = chat_engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        if backend == "openai":
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, messages: list[dict]) -> str:
        if self.backend == "vllm":
            assert self.chat_engine is not None
            assert isinstance(self.chat_engine.engine, AsyncLLMServer)
            return await self.chat_engine.chat_one_async(messages, [])
        if self.backend == "openai":
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        raise NotImplementedError(f"Unsupported backend: {self.backend}")


class ACEBenchUserSimulator:
    def __init__(self, responder: AsyncChatResponder, involved_class: list[str], language: str) -> None:
        self.responder = responder
        self.involved_class = involved_class
        self.language = language
        self.messages: list[dict] = []

    async def initialize(self, question: str) -> str:
        if self.language == "zh":
            system_prompt = SYSTEM_PROMPT_BASE_ZH if "BaseApi" in self.involved_class else SYSTEM_PROMPT_TRAVEL_ZH
            first_user = "今天有什么需要帮助的吗？"
        else:
            system_prompt = SYSTEM_PROMPT_BASE_EN if "BaseApi" in self.involved_class else SYSTEM_PROMPT_TRAVEL_EN
            first_user = "Is there anything you need help with today?"
        self.messages = [
            {"role": "system", "content": system_prompt.format(instruction=question)},
            {"role": "user", "content": first_user},
        ]
        response = await self.responder.generate(self.messages)
        self.messages.append({"role": "system", "content": response})
        return response

    def step(self, message: str) -> None:
        self.messages.append({"role": "user", "content": _remove_prefix(message)})

    async def respond(self) -> str:
        response = await self.responder.generate(self.messages)
        self.messages.append({"role": "system", "content": response})
        return response


async def run_agent_multi_step(sample: dict, args, agent_responder: AsyncChatResponder) -> dict:
    history = [{"sender": "user", "recipient": "agent", "message": sample["question"]}]
    process = []
    final_state = []
    last_instances = {}

    for _ in range(args.max_dialog_turns):
        history_text = build_history_text(history, args.scenario)
        if _ == 0 or history[-1]["sender"] == "execution":
            if args.language == "zh":
                system_prompt = MULTI_STEP_AGENT_PROMPT_SYSTEM_ZH
                user_prompt = MULTI_STEP_AGENT_PROMPT_USER_ZH.format(functions=sample["function"], history=history_text)
            else:
                system_prompt = MULTI_STEP_AGENT_PROMPT_SYSTEM_EN
                user_prompt = MULTI_STEP_AGENT_PROMPT_USER_EN.format(functions=sample["function"], history=history_text)
            response = await agent_responder.generate(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            recipient = "execution" if looks_like_function_call(response) else "user"
            history.append({"sender": "agent", "recipient": recipient, "message": response})
            if "finish conversation" in response:
                break
        else:
            outer = extract_outermost_bracket_content(history[-1]["message"]) or history[-1]["message"]
            process.append(outer.strip())
            if not looks_like_function_call(outer):
                history.append(
                    {
                        "sender": "execution",
                        "recipient": "agent",
                        "message": "Please do not ask me any questions, use the known conditions to solve the problem",
                    }
                )
                continue
            from evaluate.acebench.parser import decode_function_list

            execution_results, last_instances = execute_agent_func_call(
                func_call_list=decode_function_list(outer),
                initial_config=sample["initial_config"],
                involved_classes=sample["involved_classes"],
                model_name=args.model_alias,
                test_entry_id=sample["id"].split("_")[-1],
                language=args.language,
                scenario=args.scenario,
                acebench_root=args.acebench_root,
            )
            parsed_results = []
            for item in execution_results:
                try:
                    import json

                    parsed_results.append(json.loads(item))
                except Exception:
                    parsed_results.append(item)
            history.append({"sender": "execution", "recipient": "agent", "message": parsed_results})
            final_state = snapshot_instances(last_instances)

    end_to_end_accuracy, errors = compute_end_to_end_accuracy(final_state, sample["ground_truth"])
    process_accuracy = compute_process_accuracy(process, sample["mile_stone"], end_to_end_accuracy)
    return {
        "id": sample["id"],
        "messages": history,
        "process": process,
        "final_state": final_state,
        "data_source": sample.get("data_source", f"ACEBench/{args.scenario}"),
        "model": args.model_path,
        "metrics": {
            "end_to_end_accuracy": end_to_end_accuracy,
            "process_accuracy": process_accuracy,
        },
        "errors": errors,
    }


async def run_agent_multi_turn(
    sample: dict,
    args,
    agent_responder: AsyncChatResponder,
    user_responder: AsyncChatResponder,
) -> dict:
    user_simulator = ACEBenchUserSimulator(user_responder, sample["involved_classes"], args.language)
    init_message = await user_simulator.initialize(sample["question"])
    history = [{"sender": "user", "recipient": "agent", "message": init_message}]
    process = []
    final_state = []
    last_instances = {}

    for index in range(args.max_dialog_turns):
        last_recipient = history[-1]["recipient"]
        if last_recipient == "user":
            user_simulator.step(history[-1]["message"])
            current_user_message = await user_simulator.respond()
            history.append({"sender": "user", "recipient": "agent", "message": current_user_message})
        elif last_recipient == "agent":
            history_text = build_history_text(history, args.scenario)
            if args.language == "zh":
                system_prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH
                user_prompt = MULTI_TURN_AGENT_PROMPT_USER_ZH.format(functions=sample["function"], history=history_text)
            else:
                system_prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_EN
                user_prompt = MULTI_TURN_AGENT_PROMPT_USER_EN.format(functions=sample["function"], history=history_text)
            response = await agent_responder.generate(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            recipient = "execution" if looks_like_function_call(response) else "user"
            history.append({"sender": "agent", "recipient": recipient, "message": response})
        else:
            outer = extract_outermost_bracket_content(history[-1]["message"]) or history[-1]["message"]
            process.append(outer.strip())
            from evaluate.acebench.parser import decode_function_list

            execution_results, last_instances = execute_agent_func_call(
                func_call_list=decode_function_list(outer),
                initial_config=sample["initial_config"],
                involved_classes=sample["involved_classes"],
                model_name=args.model_alias,
                test_entry_id=sample["id"].split("_")[-1],
                language=args.language,
                scenario=args.scenario,
                acebench_root=args.acebench_root,
            )
            parsed_results = []
            for item in execution_results:
                try:
                    import json

                    parsed_results.append(json.loads(item))
                except Exception:
                    parsed_results.append(item)
            history.append({"sender": "execution", "recipient": "agent", "message": parsed_results})
            final_state = snapshot_instances(last_instances)

        if index > 1 and isinstance(history[-1]["message"], str) and "finish conversation" in history[-1]["message"]:
            break

    end_to_end_accuracy, errors = compute_end_to_end_accuracy(final_state, sample["ground_truth"])
    process_accuracy = compute_process_accuracy(process, sample["mile_stone"], end_to_end_accuracy)
    return {
        "id": sample["id"],
        "messages": history,
        "process": process,
        "final_state": final_state,
        "data_source": sample.get("data_source", f"ACEBench/{args.scenario}"),
        "model": args.model_path,
        "metrics": {
            "end_to_end_accuracy": end_to_end_accuracy,
            "process_accuracy": process_accuracy,
        },
        "errors": errors,
    }

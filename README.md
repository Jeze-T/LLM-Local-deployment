# LLM-本地部署

## 参考
>https://juejin.cn/post/7484113051483570216

## config.json配置详解

```
{
  "attention_dropout": 0,          // 注意力机制中的dropout概率，0表示不使用注意力丢弃
  "bos_token_id": 151643,          // 文本开始的特殊标记（Begin-of-Sequence）的ID
  "eos_token_id": 151645,          // 文本结束的特殊标记（End-of-Sequence）的ID
  "hidden_act": "silu",            // 隐藏层使用的激活函数（Swish/SiLU激活函数）
  "hidden_size": 896,              // 隐藏层维度（每个Transformer层的神经元数量）
  "initializer_range": 0.02,       // 权重初始化的范围（正态分布的σ值）
  "intermediate_size": 4864,       // Feed Forward层中间维度（通常是hidden_size的4-8倍）
  "max_position_embeddings": 32768,// 支持的最大上下文长度（32K tokens）
  "max_window_layers": 21,         // 使用滑动窗口注意力的最大层数（长文本优化相关）
  "model_type": "qwen2",           // 模型架构标识（用于Hugging Face库识别）
  "num_attention_heads": 14,       // 注意力头的总数（常规注意力）
  "num_hidden_layers": 24,         // Transformer层的总数量（模型深度）
  "num_key_value_heads": 2,        // 分组查询注意力中的键值头数（GQA技术，显存优化）
  "rms_norm_eps": 0.000001,        // RMS归一化的极小值（防止计算溢出）
  "rope_theta": 1000000,           // RoPE位置编码的基频参数（扩展上下文能力相关）
  "sliding_window": 32768,         // 滑动窗口注意力大小（每个token可见的上下文范围）
  "tie_word_embeddings": true,     // 是否共享输入输出词嵌入权重（常规做法）
  "torch_dtype": "bfloat16",       // PyTorch张量精度类型（平衡精度与显存）
  "transformers_version": "4.43.1",// 适配的Transformers库版本
  "use_cache": true,               // 是否使用KV缓存加速自回归生成
  "use_sliding_window": false,      // 是否启用滑动窗口注意力机制（当前禁用）
  "vocab_size": 151936             // 模型所有词表大小（支持的token总数量）
}
```

### 重点参数说明
1.长上下文支持组：
max_position_embeddings + rope_theta + sliding_window 共同实现超长上下文处理能力（32K tokens）

"rope_theta"："1000000" RoPE位置编码的基频参数 +滑动窗口机制，增强模型的长文本处理能力

2.显存优化组：
"num_key_value_heads"："2"： 分组查询注意力，减少显存占用

"torch_dtype"："bfloat16" 数据类型+use_cache：true KV缓存加速提升推理效率

AI模型本质是一个大矩阵，他由很多个参数构成，这些参数的数据类型是float32(单精度)，比float32精度更高的是float64(双精度，java里是double)，在AI模型中默认是单精度float32，这个精度有个特点是可以往下降，叫做精度的量化操作，可以把32位的精度降为16(半)位或者更低8位，最低是4位，精度变低，模型的体积会更小，运算的速度会更快，量化的意义就在于加速模型的推理能力，降低对于硬件的依赖。

现在使用的是通义千问2.5-0.5B-Instruct，这里的0.5B就代表着参数的数量，bfloat16就算每个参数所占的存储空间，模型的大小=参数的数量*每个参数所占的存储空间

3.架构特征组：
"hidden_act":"silu" 隐藏层使用现代激活函数 silu（相比ReLU更平滑）

"num_hidden_layers"："24" 深层网络结构（24层）配合intermediate_size：4864 大中间层维度增强模型容量

"num_attention_heads"："14" 较小的注意力头数平衡计算效率

4.兼容性组：
"transformers_version": "4.43.1" 明确标注Transformers库版本确保API兼容性

"model_type": "qwen2" 声明模型架构形式用于Hugging Face库识别

## generation_config.json配置详解
```
{
  "bos_token_id": 151643,       // 文本开始标记（Begin-of-Sequence）的ID，用于标识序列的起始位置
  "pad_token_id": 151643,       // 填充标记（Padding）的ID，用于对齐不同长度的输入序列（罕见地与BOS共用）
  "do_sample": true,            // 启用采样模式（若设为false则使用贪心解码，关闭随机性）
  "repetition_penalty": 1.1,    // 重复惩罚系数（>1的值会降低重复内容的生成概率）
  "temperature": 0.7,           // 温度参数（0.7为适度随机，值越高生成越多样化但也越不可控）
  "top_p": 0.8,                 // 核采样阈值（只保留累计概率超过80%的最高概率候选词集合）
  "top_k": 20,                  // 截断候选池大小（每步只保留概率最高的前20个候选token）
  "transformers_version": "4.37.0"  // 对应的Transformers库版本（兼容性保障）
}
```

### 核心参数详解
如本地化使用通义千问，可通过此文件修改文本生成的效果

1.核心采样机制组：
do_sample=true： 开启概率采样模式（关闭时为确定性高的贪心搜索）

temperature=0.7： 中等随机水平（比默认1.0更保守，适合作业型文本生成）

top_p=0.8 + top_k=20： 组合采样策略（双重约束候选词的质量）

2.文本优化控制组：
repetition_penalty=1.1： 温和的重复惩罚参数（值越大越压制重复用词）

bos_token_id=pad_token_id： 特殊标记共享设计（注意：这与常规模型设计不同，可能用于处理可变长度输入的特殊场景）

3.工程兼容性：
明确标注transformers_version，提示用户需对应版本的库才能正确解析配置

## tokenizer_config.json
```
{
  "add_bos_token": false,          // 是否自动添加文本起始符（BOS），关闭表示输入不额外增加开始标记
  "add_prefix_space": false,       // 是否在文本前自动添加空格（用于处理单词边界，该项对中文无影响）
  "bos_token": null,               // 不设置专门的BOS标记（模型的起始序列由对话模板控制）
  "chat_template": "Jinja模板内容",// 角色对话模板规范（定义：角色标记+工具调用格式+消息边界切割规则）
  "clean_up_tokenization_spaces": false,  // 是否清理分词产生的空格（保留原始空格格式）
  "eos_token": "<|im_end|>",       // 文本终止标记（用于标记每个对话轮次/整体生成的结束）
  "errors": "replace",             // 解码错误处理策略（用�符号替换非法字符）
  "model_max_length": 131072,      // 最大处理长度（131072 tokens约等于128K上下文容量）
  "pad_token": "<|endoftext|>",    // 填充标记（用于批量处理的序列对齐）
  "split_special_tokens": false,   // 禁止分割特殊标记（确保类似<|im_end|>保持整体性）
  "tokenizer_class": "Qwen2Tokenizer",    // 专用分词器类型（特殊字词分割规则实现）
  "unk_token": null                // 无专用未知标记（通过errors策略处理未知字符）
}
```

### 核心参数详解
1.长上下文支持：
"model_max_length": 131072 显式描述支持128K超长文本处理

"clean_up_tokenization_spaces": false 维护原始空格格式确保长文本连贯性

2.对话格式控制：
chat_template 定义多模态对话逻辑：

eos_token 作为终止符贯穿全对话流程控制<|im_start|>/<|im_end|> 标记划分角色边界（system/user/assistant/tool）

<tool_call> XML块规范函数调用输出格式

之前的大模型是通过MaxLength来控制文本的最大输出，现在的大模型需要根据问题的答案长短来控制文本的输出，所以他这个模型使用eos_token定义的终止符，来做提前终止。这些特殊的终止符还有一个作用是与大模型的对话模板组合控制生成内容。

3.中文优化特性：
"add_prefix_space": false 避免英文分词策略对中文处理干扰

"unk_token": null 结合 "errors": "replace" 实现对中文生僻字的容错处理

"tokenizer_class": "Qwen2Tokenizer" 专用Qwen2Tokenizer含有中文高频词汇的特殊分割规则

## 加载方式
transformer、Ollama、vLLM、LMDeploy
### transformer
transformer平台它推理模型的性能很低，它不是一个专门用来跑模型的一个框架。所以一般在真实使用的时候，不会选择用这种方式来调用大模型。
```
#使用transformer加载qwen模型
from transformers import AutoModelForCausalLM,AutoTokenizer

DEVICE = "cuda"

#加载本地模型路径为该模型配置文件所在的根目录
model_dir = "/mnt/workspace/LLM/Qwen/Qwen2.5-0.5B-Instruct"

#使用transformer加载模型
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

#调用模型
#定义提示词
prompt = "你好，请介绍下你自己。"
#将提示词封装为message
message = [{"role":"system","content":"You are a helpful assistant system"},{"role":"user","content":prompt}]
#使用分词器的apply_chat_template()方法将上面定义的消息列表进行转换;tokenize=False表示此时不进行令牌化
text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)

#将处理后的文本令牌化并转换为模型的输入张量
model_inputs = tokenizer([text],return_tensors="pt").to(DEVICE)

#将数据输入模型得到输出
response = model.generate(model_inputs.input_ids,max_new_tokens=512)
print(response)

#对输出的内容进行解码还原
response = tokenizer.batch_decode(response,skip_special_tokens=True)
print(response)
```

### Ollama
#### 1.Ollama 是什么？
Ollama 是一款开源的、专注于在本地设备上运行大型语言模型（LLM）的工具。它简化了复杂模型的部署和管理流程，允许用户无需依赖云端服务或网络连接，即可在自己的电脑、服务器甚至树莓派(微型单板计算机)等设备上体验多种先进的 AI 模型（如 Llama3、Mistral、Phi-3 等）。其核心目标是让本地化运行 LLM 变得低成本、高效率、高隐私。

#### 2.核心功能与特点
##### 1. 本地化部署
隐私与安全： 所有模型运行和数据处理均在本地完成，适合需保护敏感信息的场景（如企业数据、个人隐私）。
离线可用： 无需互联网连接，可在完全断开网络的设备使用。
资源优化： **支持量化模型（如 GGUF 格式）**，大幅降低内存占用（某些小型模型可在 8GB 内存的设备运行）。

##### 2. 丰富的模型生态
主流模型支持： Llama3、Llama2、Mistral、Gemma、Phi-3、Wizard、CodeLlama 等。
多模态扩展： 通过插件支持图像识别、语音交互等扩展功能。
自定义模型： 用户可上传自行微调的模型（兼容 PyTorch、GGUF、Hugging Face 格式）。

##### 3. 开箱即用
极简安装： 支持 macOS、Linux（Windows 通过 Docker/WSL），仅需一行命令完成安装。
统一接口： 提供命令行和 REST API，方便与其他工具（编程接口、Chat UI 应用）集成。

##### 4. 多样化应用场景
开发调试： 快速构建本地 LLM 驱动的应用原型。
企业私有化： 内部知识库问答、文档分析等场景。
研究与教育： 低成本教学 AI 模型原理与实践。

#### 3.使用Ollama部署大模型
##### 1.Conda创建单独的环境
###### 1.创建ollama环境
阿里云魔塔社区免费给的服务器每次开启需要重新激活conda：source /mnt/workspace/miniconda3/bin/activate
conda create -n ollama
###### 2.激活ollama环境
查看conda环境 ：conda env list
激活ollama环境: conda activate ollama
前面由base切换到了ollama环境，切换成功

##### 2.下载ollama的Linux版本
###### 1.进入官网
https://ollama.com/download/linux

复制命令curl -fsSL https://ollama.com/install.sh | sh

###### 2.执行命令
curl -fsSL ollama.com/install.sh | sh

但是这个玩意下载的时候很慢，这东西网络很不稳定，就是有有一段时间它的下载速度会比较快，有一段时间它下载的速度会非常慢。甚至于有一段时间它会提示你这个网络连接错误，网络连接错误。如果出现了这种情况，一般的解决方法就是最完美的解决方法是在你的服务器上面去挂个梯子，但是这个挂梯子不一定有效。所以说这个东西最大的困难就在于安装起来得靠运气。
**(有在别的博主那里学到，可以ctrl+C暂停，再运行会快很多)**
###### 3.下载安装包安装(推荐)
1.登录Github
找到ollama的Releases 或者直接跳转开始下载:
https://github.com/ollama/ollama/releases/download/v0.6.2/ollama-linux-amd64.tgz
2.选择ollama-linux-amd64.tgz
###### 4.github下载加速
https://github.moeyy.xyz/ 但是实测下来用不了因为文件1.6个G太大了会变成下图这样，老老实实慢慢下载

###### 5.上传文件到服务器并启动
这是魔塔社区提供的SSH连接服务器的方式：[ (https://help.aliyun.com/zh/pai/user-guide/dsw-direct-connection?spm=a2c4g.11186623.0.0.18133b46uFBup2)](https://help.aliyun.com/zh/pai/user-guide/dsw-direct-connection?spm=a2c4g.11186623.0.0.18133b46uFBup2)

不想重新搞了，我选择用我之前的基石智算服务器xftp上传。

上传完成后解压到usr目录不然普通用户默认无权访问，同时该路径不在系统的 PATH 环境变量中，执行： sudo tar -C /usr -xzf ollama-linux-amd64.tgz

如果出现了：-bash: sudo: command not found

执行这两个命令:apt-get update apt-get install sudo

启动ollama服务：ollama serve
出现下图内容为启动成功：

msg="Listening on 127.0.0.1:11434 (version 0.6.2) "

**服务开启后默认端口是：11434**

服务启动后，不要关闭这个窗口，另开一个窗口运行代码(重要，关了服务就没了)

可使用 ollama -v 查看版本
使用 ollama list 查看模型，现在是没有的

###### 6.ollama中添加模型
1.进入ollama官网找到千问2.5
2.选择0.5b版本，复制命令并在服务器执行   ollama run qwen2.5:0.5b

ollama run qwen2.5:0.5b 执行后就会从ollama官网拉取这个模型，拉取下来后就可以直接使用。

###### 有点胡说八道，知识库没更新。有几个问题：

1.前面下载了一个千问的模型，为啥不能直接跑？

因为ollama，他只能支持GGUF格式的模型，所以如果要跑ollama的模型，只有两个选择。第一个选择就是从它的官网上面去找模型，然后进行下载。第二个选择是，使用魔塔社区model scope的这个模型库去搜索GGUF格式的模型。

2.什么是GGUF？

一般指的是量化后的模型(阉割版)，比正常的模型小一点，量化的缺点就算模型的效果会会变差了。

所以这个水平其实不是千问0.5B真实的水平，因为它是被量化之后的水平。但是量化带来的优势就在于模型的文件会变得更小，推理速度会更快，对于硬件的要求会更低。所以说ollama针对的是个人用户，它不针对于企业。

当然ollama也可以跑不量化的模型

###### 7.使用代码调用ollama中的千问
目前的大模型推理框架里面，它的这个访问地址的接口协议都是OpenAI API的协议OpenAI API的协议，我们可以使用OpenAI API的风格来调用这个模型，调用ollama的这个模型。

model="qwen2.5:0.5b",需要使用ollama list 查看模型，用哪个写哪个：
环境中是没有openAi 的服务的执行：pip install openai

```
#使用openai的API风格调用ollama
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1/",api_key="ollamaCall")

chat_completion = client.chat.completions.create(
    messages=[{"role":"user","content":"你好，请介绍下你自己。"}],model="qwen2.5:0.5b"
)
print(chat_completion.choices[0])
```

### vLLM
#### 1.vLLM是什么
vLLM 是一个专为大语言模型（LLM）推理和服务设计的高效开源库，由加州大学伯克利分校等团队开发。它通过独特的技术优化，显著提升了模型推理的速度和吞吐量，特别适合**高并发、低延迟**的应用场景。

#### 2.核心技术：PagedAttention
**内存管理优化**： **借鉴操作系统的内存分页思想，将注意力机制的键值（KV Cache）分割成小块（分页），按需动态分配显存**。
**解决显存碎片问题**： 传统方法因显存碎片导致利用率低（如仅60-70%），**PagedAttention使显存利用率达99%以上，支持更长上下文并行处理**。
性能提升：**相比Hugging Face Transformers，吞吐量最高提升30倍**。

#### 3. 核心优势
极高性能： **单GPU支持每秒上千请求（如A100处理100B参数模型），适合高并发API服务**。
全面兼容性： 支持主流模型（如Llama、GPT-2/3、Mistral、Yi），无缝接入Hugging Face模型库。
简化部署： 仅需几行代码即可启动生产级API服务，支持动态批处理、流式输出等。
开源社区驱动： 活跃开发者迭代优化，支持多GPU分布式推理。

#### 4.使用vLLM
##### 1.进入官网地址
[vllm.hyper.ai/docs/gettin…](https://vllm.hyper.ai/docs/getting-started/installation)

**vLLM对环境是有要求的**:
**操作系统：Linux**
**Python：3.8 - 3.12**
GPU：计算能力 7.0 或更高（例如 V100、T4、RTX20xx、A100、L4、H100 等，显存16GB以上的基本都可以）
##### 2.创建vLLM的Conda环境
阿里云魔塔社区免费给的服务器每次开启需要重新激活conda：source /mnt/workspace/miniconda3/bin/activate

这个**vLLM他基于CUDA的，他与CUDA的版本是挂钩**的，所以万万不能在base环境装vLLM，因为一旦你在base环境上面装了之后的话，只要你base环境的CUDA版本不是12.11的版本，它就会把你的CUDA给你卸载了，装它对应的这个版本。目前的**vLMM只支持两个版本，一个是CUDA12.1，另外一个是CUDA11.8**，其他的他不支持，所以这个玩意儿跟CUDA挂钩。

conda create -n vLLM python==3.12 -y


##### 3.激活vLLM的Conda环境
conda activate vLLM
从base切换到vLLM成功。

##### 4.执行安装vLLM
pip install vllm

##### 5.开启vLLM服务
vllm serve + 模型绝对路径
vllm serve /mnt/workspace/LLM/Qwen/Qwen2.5-0.5B-Instruct

产生这个错误是因为服务器第一次装的时候在root环境里面，新建的vLLM环境里面缺失modelscope的依赖包 ,所以执行pip install modelscope 安装即可

启动成功
**服务的端口是8000**

##### 6.使用代码调用vLLM服务中的千问
```
#多轮对话
from openai import OpenAI

#定义多轮对话方法
def run_chat_session():
    #初始化客户端
    client = OpenAI(base_url="http://localhost:8000/v1/",api_key="vLLMCall")
    #初始化对话历史
    chat_history = []
    #启动对话循环
    while True:
        #获取用户输入
        user_input = input("用户：")
        if user_input.lower() == "exit":
            print("退出对话。")
            break
        #更新对话历史(添加用户输入)
        chat_history.append({"role":"user","content":user_input})
        #调用模型回答
        try:
            chat_complition = client.chat.completions.create(messages=chat_history,model="/mnt/workspace/LLM/Qwen/Qwen2.5-0.5B-Instruct")
            #获取最新回答
            model_response = chat_complition.choices[0]
            print("AI:",model_response.message.content)
            #更新对话历史（添加AI模型的回复）
            chat_history.append({"role":"assistant","content":model_response.message.content})
        except Exception as e:
            print("发生错误：",e)
            break
if __name__ == '__main__':
    run_chat_session()
```
调用成功

### LMDeploy
#### 1.LMDeploy是什么
LMDeploy 是一个由 **上海人工智能实验室（InternLM团队）** 开发的工具包，专注于大型语言模型（LLM）的压缩、部署和服务化。
它的目标是帮助开发者和企业更高效地在实际场景中应用大模型（如百亿到千亿参数规模模型），解决**高计算资源消耗和延迟**等问题。

#### 2.核心特点
##### 1. 高效推理优化
**模型压缩**：**支持 KV8 量化和 Weight INT4 量化**，显著降低显存占用（最高可减少 50%），提升推理速度。
**持续批处理（Continuous Batch）** ：**动态合并用户请求的输入**，提高 GPU 利用率。
**页显存管理**：通过**类似操作系统的显存管理策略**，进一步提升吞吐量。
##### 2. 多后端支持
内置**高性能推理引擎 TurboMind，支持 Triton 和 TensorRT-LLM 作为后端**，适配本地和云端部署。
兼容 Transformers 模型结构，轻松部署 Hugging Face 等平台的预训练模型。
##### 3. 多模态扩展
支持视觉-语言多模态模型（如 LLaVA），实现端到端的高效推理。
##### 4. 便捷的服务化
提供 RESTful API 和 WebUI，支持快速搭建模型服务，适配云计算、边缘计算等场景。

#### 3.核心技术
##### 1.KV Cache 量化
减少推理过程中键值（Key-Value）缓存的内存占用。
**vLLM是内存分页，LMDeploy是内存量化**

##### 2.W4A16 量化
将**模型权重压缩至 INT4 精度**，保持精度损失极小（<1%）。
**外加模型量化**

##### 3.深度并行化
利用**模型并行、流水线并行**等技术，支撑千亿级模型的**分布式部署**。
**模型并行、请求处理并行**

#### 4.使用LMDeploy
##### 1.进入官网
[lmdeploy.readthedocs.io/zh-cn/lates…](https://lmdeploy.readthedocs.io/zh-cn/latest/index.html)

##### 2.创建LMDeploy的Conda环境
conda create -n lmdeploy python==3.12 -y

##### 3.激活LMDeploy的Conda环境
conda activate lmdeploy

##### 4.执行安装LMDeploy
pip install lmdeploy

##### 5.开启LMDeploy服务
lmdeploy serve api_server + 模型的全路径

执行：lmdeploy serve api_server /root/LLM/Qwen/Qwen2.5-0.5B-Instruct ，报了ModuleNotFoundError: No module named 'partial_json_parser'，

可能是因为lmdeploy与python3.12版本的兼容问题，执行 pip install partial-json-parser 命令

重新执行：lmdeploy serve api_server /root/LLM/Qwen/Qwen2.5-0.5B-Instruct

启动成功

##### 6.使用代码调用LMDeploy服务中的千问
LMDeploy 比vLLM在**显存**的优化上要强很多，这个框架对于**模型的量化**支持力度比vLLM要更全要更全。它支持目前的两种主流量化，一种是离线的量化，一种是在线的量化。
**页显存管理**

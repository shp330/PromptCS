import math
import torch.nn as nn
import torch
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaConfig, RobertaModel, PreTrainedModel
from typing import List, Tuple


class PromptCS(nn.Module):
    """
    Attributes:
        mode (str): finetune 或 PromptCS
        model (PreTrainedModel): prompt agent 底层使用的模型
        loss_fct (nn.CrossEntropyLoss): 损失函数；forward(input: Tensor, target: Tensor)
        max_target_length (int): 生成的答案最大长度
        max_code_length (int): 输入的代码使用的最大有效长度
    """
    mode: str
    model: PreTrainedModel
    loss_fct: nn.CrossEntropyLoss
    max_target_length: int
    max_code_length: int

    def __init__(self, args, device, template: List[int]):
        """
        :param args:
            args.mode: finetune | PromptCS
            args.max_target_length
            args.max_code_length
            args.prompt_encoder_type: transformer | lstm
        :param device:
        :param template:
            表示 prompt 模板的结构。每个元素表示对应 segment 的长度（即多少个 token）。
            模板：[CLS] + 文本 + [prompt] + 文本 + [SEP]
                 # 1个[CLS], 5个prompt token, 1个[SEP], 文本长度为0（由输入决定）
                 # 0：表示由原始输入文本占据的位置（长度可变）
                 # >0：表示可训练的 soft prompt 或 hard prompt 占据的位置（长度固定）
                 self.template = [1, 0, 5, 0, 1]
        :return:
        """
        super(PromptCS, self).__init__()
        self.args = args
        self.mode = args.mode
        self.device = device
        self.use_lm_finetune = False

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        if args.mode == 'finetune':
            self.use_lm_finetune = True
            template = [0, 0]
        self.template = template

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        self.pad_token_id, self.sep_token_id, self.eos_token_id, self.unk_token_id = self.get_special_token_id()
        # prompt_tokens、sep_tokens、eos_tokens 必须为list，否则后续 [] * int 和 [] + [] 运算会报错
        self.prompt_tokens = [self.pseudo_token_id]
        self.sep_tokens = [self.sep_token_id]
        self.eos_tokens = [self.eos_token_id]

        # load pre-trained model
        self.model = create_model(self.args, self.use_lm_finetune)
        self.model = self.model.to(self.device)
        # 深度学习中常见的"模型参数梯度控制"操作，通常出现在使用 PyTorch 的场景中
        # self.model.parameters() 返回模型中所有可训练参数（如权重、偏置）的迭代器,
        # 这些参数是 torch.nn.Parameter 类型，属于 torch.Tensor 的子类。
        for param in self.model.parameters():
            ##### param.requires_grad 是 Tensor 的一个布尔属性，决定该参数是否需要计算梯度。
            # 默认情况下，大多数参数的 requires_grad=True
            # True：在反向传播（loss.backward()）时会计算梯度，可用于优化器更新,如 optimizer.step()。
            # False：不计算梯度，不参与训练。
            #
            ##### self.use_lm_finetune，表示是否对语言模型进行微调（fine-tune）。
            # 如果是 True：保留梯度，允许模型参数更新（即进行微调）。
            # 如果是 False：冻结模型参数，不更新（通常用于仅使用模型推理，或只训练其他部分，如适配层）。
            #
            # 根据 self.use_lm_finetune 的值，决定是否冻结整个模型的参数。
            # self.use_lm_finetune = True：所有参数 requires_grad = True, 模型可以被微调（训练）
            # 如果为 False：所有参数 requires_grad = False, 模型被冻结，不参与梯度更新，仅用于前向推理
            param.requires_grad = self.use_lm_finetune
        self.embeddings = get_embedding_layer(self.args, self.model)

        # load prompt encoder
        # embedding_dim 嵌入向量的维度。用途：
        # 1、后续层（如 LSTM、Transformer）需要知道输入的维度。
        # 2、用于初始化可训练的 soft prompt 向量、LoRA 适配器等
        self.hidden_size = self.embeddings.embedding_dim
        #### 计算 soft prompt 的总长度
        # 整个 prompt 模板中所有固定 token 的总数（包括 [CLS], [SEP], [MASK], soft prompt 等）。
        ## 作用：
        # 1、在 P-tuning 中，spell_length 表示需要可训练的连续向量个数（soft prompt 长度）。
        # 2、用于初始化一个形状为[batch_size, spell_length, hidden_size] 的可训练嵌入矩阵。
        self.spell_length = sum(self.template)

        if args.mode == 'PromptCS':
            self.prompt_agent = PromptAgent(self.template, self.hidden_size, self.tokenizer, self.device, args)
            self.prompt_agent = self.prompt_agent.to(self.device)

        self.max_target_length = args.max_target_length
        self.max_code_length = args.max_code_length
        #### 自然语言处理（NLP）任务中损失计算相关的核心部分，常见于序列生成模型（如语言模型、机器翻译、文本摘要）或分类任务中
        ## 将模型输出的原始 logits 转换为 log-probabilities，便于后续计算负对数似然（NLL）损失。
        #
        ## 它对输入张量进行 Log-Softmax 操作：
        #  1、先计算 Softmax：将原始 logits 转换为概率分布（值在 0~1 之间，总和为 1）。
        #  2、再取对数（log）：得到 log-probabilities。
        #
        ## dim=-1 表示在最后一个维度上进行操作。
        # 对于语言模型输出：logits.shape = [batch_size, sequence_length, vocab_size],
        # dim=-1 就是对 vocab_size 维度做 LogSoftmax，即每个 token 对应的词表上所有词的概率分布
        #
        ## 为什么用 LogSoftmax 而不是 Softmax？
        #  1、数值稳定性：直接计算 log(softmax(x)) 容易下溢或上溢。
        #  2、PyTorch 的 LogSoftmax 使用了稳定的算法（如减去最大值）来避免数值问题
        #
        # lsm：LogSoftmax 缩写
        self.lsm = nn.LogSoftmax(dim=-1)

        #### PyTorch 提供的交叉熵损失函数，用于分类任务，等价于 LogSoftmax + NLLLoss 的组合。
        # 输入：原始 logits（不需要先做 softmax）。
        # 输出：标量损失值。
        #
        ## ignore_index=self.pad_token_id，指定一个 token ID，在计算损失时忽略该 token。
        #  self.pad_token_id 通常是 [PAD] 的 ID（如 0）。
        #  在批处理中，不同长度的序列会被 padding 到相同长度，这些 padding 位置不应参与损失计算。
        #
        ## reduction='sum'，控制损失的归约方式；使用 'sum' 意味着：总损失是所有非 padding 位置损失值的总和。
        #  'mean'：取平均（默认）
        #  'sum'：求和
        #  'none'：不归约，返回每个样本的损失
        #
        ## 为什么用 'sum'
        #  1、在某些训练策略中（如梯度累积、动态 batch size），使用 sum 比 mean 更容易控制学习率和梯度尺度。
        #  2、特别是在序列到序列任务中，按 sum 计算可以更好地反映整体生成质量。
        #
        # fct：Function 缩写
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')

    def get_special_token_id(self):
        """
        获取所使用模型（args.model_name_or_path）的特殊的token id
        """
        pad_token_id, sep_token_id, eos_token_id, unk_token_id = None, None, None, None
        model_name = self.args.model_name_or_path.lower()
        if 'starcoder' in model_name:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            sep_token_id = self.vocab['<fim_middle>']
            eos_token_id = self.tokenizer.eos_token_id
            unk_token_id = self.tokenizer.unk_token_id
        elif 'polycoder' in model_name:
            pad_token_id = self.vocab['<|padding|>']
            sep_token_id = self.vocab['<|separator|>']
            eos_token_id = self.vocab['<|endoftext|>']
            unk_token_id = self.vocab['<|padding|>']
        elif 'codegen' in model_name:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            sep_token_id = self.vocab['//']
            eos_token_id = self.tokenizer.eos_token_id
            unk_token_id = self.tokenizer.unk_token_id

        return pad_token_id, sep_token_id, eos_token_id, unk_token_id

    def embed_input(self, queries_token_ids: Tensor) -> Tensor:
        """
        根据 mode（PromptCS | Finetune）获取输入嵌入向量

        Args:
            queries_token_ids(Tensor): Token id. 大小为 B x L x * 的张量. B 为批次大小；L 为填充后的序列长度

        Returns:
            Tensor: 形状为 (batch_size, sequence_length, hidden_size) 的嵌入张量
        """
        if self.mode == 'PromptCS':
            return self.cstuning_embed_input(queries_token_ids)
        else:
            return self.finetune_embed_input(queries_token_ids)

    def finetune_embed_input(self, queries_token_ids: Tensor):
        """
        获取微调模式的输入嵌入向量

        Args:
            queries_token_ids: 包含经过填充的输入 token 的 id 的张量，形状为 (batch_size, sequence_length)，
                 其中包含用于 prompt 的伪 token（如 `self.pseudo_token_id`）

        Returns:
            所有输入 token 的嵌入向量张量

        """
        return self.embeddings(queries_token_ids)

    def cstuning_embed_input(self, queries_token_ids: Tensor):
        """
        生成 PromptCS 模式的输入嵌入向量，其中伪 token 位置由可学习的 prompt embeddings 替换。
        该方法将输入序列中的特定 token（如 [PROMPT]）替换为可训练的 prompt embeddings，
        实现“软提示”（soft prompt）注入，用于提示微调（prompt tuning）。

        Args:
            queries_token_ids:  包含经过填充的输入 token 的 id 的张量，形状为 (batch_size, sequence_length)，
                 其中包含用于 prompt 的伪 token（如 `self.pseudo_token_id`）。

        Returns:
            Tensor: 形状为 (batch_size, sequence_length, hidden_size) 的嵌入张量，
            其中伪 token 位置已被可学习的 prompt embeddings 替换。学习后的 prompt embeddings

        Example:
            输入:
                queries_token_ids = [
                    [101, 50000, 50000, 7592, 2087, 102],  # [CLS] [P] [P] "hello" "world" [SEP]
                    [101, 50000, 50000, 1996, 2000, 102]   # [CLS] [P] [P] "how" "are" [SEP]
                ]

                # 形状: (bz, spell_length), spell_length 是伪 prompt 的长度
                # 每一行表示一个样本中所有 [PROMPT] token 在序列中的位置。后续用于替换这些位置的 embedding。
                blocked_indices =[
                  [1, 2],  # 第 0 个样本中，prompt 在位置 1 和 2
                  [1, 2]   # 第 1 个样本中，prompt 在位置 1 和 2
                ]

            输出: [CLS] + (learned_prompt_embeds) + [hello] [world] [SEP]


        Notes:
            输入序列里会插入一些特殊的 “伪 token” 占位符, 这些占位符在原始词表里没有实际意义，我们不希望用它的 embedding,
            所以先把它们替换成 UNK token，让 embedding 层用 UNK 的向量作为临时值,
            之后再用 self.prompt_agent() 生成的可学习 prompt embedding 覆盖这些位置
        """
        # batch size
        bz = queries_token_ids.shape[0]
        # 复制输入 token ids，用于后续嵌入查找
        queries_for_embedding = queries_token_ids.clone()
        # 将伪 token ID 替换为 [UNK]，避免其对原始 embedding 干扰
        # 这样在查嵌入表时，这些位置会使用 [UNK] 的 embedding（或后续被替换）
        #
        ## 语法：
        # 找到 queries_token_ids 中所有值等于 self.pseudo_token_id 的位置，
        # 然后把 queries_for_embedding 中对应位置的值替换为 self.unk_token_id。
        #
        ## 这在Prompt Tuning、Prefix Tuning 等场景中经常用来：
        # 先把 “伪 token” 位置替换成 UNK（未知词），后面再用自定义的 embedding 覆盖这些位置
        queries_for_embedding[(queries_token_ids == self.pseudo_token_id)] = self.unk_token_id
        # 获取原始嵌入：先用替换后的 token ids 查嵌入表
        raw_embeds = self.embeddings(queries_for_embedding)  # (bz, seq_len, hidden_size)

        # 找出所有伪 token 的位置（即 prompt 应插入的位置）, 并提取其在序列中的列索引
        # 注意：假设每个样本有 self.spell_length 个连续的伪 token
        # 结果: (bz, spell_length)，每行是该样本中 prompt token 在序列中的位置索引
        blocked_indices = (
            (queries_token_ids == self.pseudo_token_id)
            .nonzero()
            .reshape((bz, self.spell_length, 2))[:, :, 1]
        )

        # 获取学习后的 prompt embeddings
        # 对象调用：调用 prompt_agent 的 forward 方法，返回 (spell_length, hidden_size)
        # replace_embeds 列表的长度为虚拟prompt token 的长度
        replace_embeds = self.prompt_agent()
        for bidx in range(bz):  # bidx 批索引，代笔第几个样本
            for i in range(self.prompt_agent.spell_length):
                # 更新 prompt embedding，将原始嵌入的 prompt token 的位置更新为对应的学习后的prompt embedding
                # 注意：每个样本特定位置的 prompt token 的学习后的嵌入相同
                # blocked_indices[bidx, i] 表示 第 bidx 个样本的第 i 个 prompt 位置
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, x_h: str, x_t: str = None) -> Tuple[torch.LongTensor, int]:
        """
        将输入文本转为token id，并构造 Prompt-based 输入；最终结构为：[P][P] head [P] [SEP] target [EOS]
        在代码-文本匹配任务中，x_t 代表代码，x_t 代表描述

        Args:
            x_h: 头实体文本
            x_t: 目标实体文本

        Returns:
            Tuple[torch.LongTensor, int]:
                input_ids: tokenizer 编码后的 token id 序列
                len(left): idx，用于在 forward 中标记目标序列的起始位置（后面用来切分 logits）
        """
        '''
        self.tokenizer.tokenize(x_h) 对头实体分词，并取最大代码长度
        tokenizer.convert_tokens_to_ids 转为 token ID
        比如：
            prompt_tokens = [1000]（自定义虚拟 token ID）
            template = [2, 1]                → 前面加 2 个 prompt token，后面加 1 个
            x_h = "def add(a,b): return a+b" → tokenize 后取前 N 个
            left = [1000,1000] + [ids of x_h] + [1000]
        左右：构建包含 soft prompt 和头实体的左侧上下文。    
        '''
        left = self.prompt_tokens * self.template[0] + self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(x_h)[:self.max_code_length]
        ) + self.prompt_tokens * self.template[1]

        if x_t is not None:
            '''
            对尾实体或目标文本分词并截取最大长度后转为token id，之后添加结束符
            作用：
                1、训练时：提供监督信号（完整序列）
                2、推理时：x_t=None，right=[]，模型需自回归生成
            '''
            right = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(x_t)[:self.max_target_length]
            ) + self.eos_tokens
        else:
            right = []

        input_ids = left + self.sep_tokens + right

        return torch.LongTensor(input_ids), len(left)

    def prepare_inputs(self, batch_input_ids: List[torch.LongTensor]) -> Tuple[
        torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        输入预处理
            使用 pad_token_id 对输入 token ids 进行填充，并将"批量维度"置于第一个；
            生成注意力掩码；
            获取输入嵌入向量

        Args:
            batch_input_ids (List[LongTensor]): 批次所有输入的 token ids 列表

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
                - batch_input_ids (torch.LongTensor): 填充后的输入token ids Tensor <br>
                - inputs_embeds (torch.LongTensor): 输入嵌入 <br>
                - attention_mask (torch.LongTensor): 注意力掩码
        """
        batch_input_ids = pad_sequence(batch_input_ids, True, padding_value=self.pad_token_id).long().to(self.device)

        # 生成注意力掩码
        # 作用：Transformer 模型在计算注意力时默认会关注序列所有位置，
        #      如果不屏蔽 padding，模型会把 padding token 当成正常 token 参与计算，影响结果，
        #      attention_mask 会在计算时屏蔽 padding 位置的影响
        #
        # 生成一个和 batch_input_ids 形状相同的布尔 tensor
        # True 表示该位置是真实 token（需要被注意力机制关注）
        # False 表示该位置是 padding（需要在注意力计算中被忽略）
        attention_mask = batch_input_ids != self.pad_token_id
        inputs_embeds = self.embed_input(batch_input_ids)

        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.mode != 'finetune':
            inputs_embeds = inputs_embeds.half()
            attention_mask = attention_mask.half()

        return batch_input_ids, inputs_embeds, attention_mask

    def forward(self, x_hs: List[str] = None, x_ts: List[str] = None) -> List[str]:
        """
        训练模型，或生成答案
        Args:
            x_hs: 头实体序列（head sequence）通常是自然语言描述或实体名
            x_ts: 目标实体序列（target sequence）可选

        Returns:
            List[str]: 批次所有样本的答案

        Notes:
            这种设计支持两种模式：
                训练模式：同时传入 x_hs 和 x_ts（有目标，计算 loss）
                推理模式：只传入 x_hs（无目标，生成答案）
        """
        # batch size
        bz = len(x_hs)
        # 训练模式，计算 loss
        if x_ts is not None:
            # 同一批次样本所有输入的ids
            batch_input_ids, sum_idx, ext_inputs = [], [], []
            for i in range(bz):
                # 构造模板化的 prompt 模型输入：prompt + input + target
                # input_ids：tokenizer 编码后的 token id 序列
                # idx：记录 target 开始位置（后面用来切分 logits），用于后续计算 loss
                input_ids, idx = self.get_query(x_hs[i], x_ts[i])
                batch_input_ids.append(input_ids)
                sum_idx.append(idx)

            batch_input_ids, inputs_embeds, attention_mask = self.prepare_inputs(batch_input_ids)

            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            # logits 形状：(batch_size, sequence_length, vocab_size)
            logits = output.logits
            loss = None

            for i in range(bz):
                # input 构造：prompt + [SEP] + target
                # [SEP] token 的索引；
                idx = sum_idx[i]
                # shift_logits 取 [idx:-1, :]；即，从 [SEP] 位置开始，其值是下一个 token 的分数;即，模型读完 [SEP] 后开始生成 response
                # shift_labels 取 [idx + 1:]；即，从 [SEP] 的下一个 token 开始，这样 logits[idx] 预测的是 labels[idx+1]
                # 从目标序列起始位置到末尾（不包含最后一个 token）
                #
                ## logits[i]: 第 i 个样本的输出 logits，形状为 (seq_len, vocab_size)
                ## logits[i][idx:-1, :]: 取从 idx 到倒数第二个 token 的 logits
                ## .contiguous(): 确保内存连续，为后续 view 做准备
                shift_logits = logits[i][idx:-1, :].contiguous()

                # 获取 target 部分的真实标签。目标序列向后偏移一位（语言模型的标准做法）
                # batch_input_ids[i][idx + 1:]: 从 idx+1 开始到最后的真实 token ID
                shift_labels = batch_input_ids[i][idx + 1:].contiguous()

                if loss is None:
                    # CrossEntropyLoss, input: Tensor, target: Tensor
                    # shift_logits.view(-1, vocab_size)：展平成 (target_len, vocab_size)
                    # shift_labels.view(-1)：展平成 (target_len,)
                    loss = self.loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                else:
                    loss += self.loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

            loss = loss / bz

            return loss
        # 自回归生成（autoregressive generation）流程，
        # 用于在给定前缀（如 prompt 或 head）后，生成目标文本（如 answer 或 response）
        else:
            # tmp_idx 当前正在生成的 token 的索引，每生成一个token，则往前移动 1
            queries, sum_idx, tmp_idx = [], [], []
            for i in range(bz):
                input_ids, idx = self.get_query(x_h=x_hs[i])
                queries.append(input_ids)
                sum_idx.append(idx)
                tmp_idx.append(idx)

            for _ in range(self.max_target_length):
                batch_input_ids, inputs_embeds, attention_mask = self.prepare_inputs(queries)

                output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

                logits = output.logits
                for i in range(bz):
                    idx = tmp_idx[i]
                    tmp_idx[i] += 1
                    next_token_logits = logits[i, idx:idx + 1, :]
                    # 返回值最大的元素的 values, indices
                    _, next_token = torch.max(next_token_logits, dim=1)

                    queries[i] = torch.cat(
                        [queries[i].to(self.device), next_token],
                        dim=0
                    )

            answers = []
            for i in range(bz):
                idx = sum_idx[i]
                t = queries[i][idx + 1:]
                t = t.tolist()
                if self.eos_token_id in t:
                    t = t[:t.index(self.eos_token_id)]
                words = self.tokenizer.decode(t).replace('\n', '')
                answers.append(words)

            return answers


class PromptAgent(torch.nn.Module):
    """
    torch.nn.Module 实现了 __call__ 方法，它会自动调用自定义的 forward 方法。
    PyTorch 官方明确建议：始终使用 model(input)，而不是 model.forward(input)
    执行 output = model(input) 时，实际执行的是：
        output = model.__call__(input)  # Module 的 __call__ 方法
    而 __call__ 内部会调用：
        output = model.forward(input)  # 自定义的 forward 方法

    Attributes:
        spell_length(int): prompt template 的长度
        embedding(torch.nn.Embedding): 代表 prompt token 序列的嵌入
        seq_indices(torch.LongTensor): 代笔 prompt token 索引的张量
    """

    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.embed_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        #### ent embedding
        ## 记录每个可训练或预测区域的长度。
        #
        #  基于模板的实体嵌入（Entity Embedding）或完形填空（Cloze）任务中构建序列结构的一部分，
        #  常见于 Prompt-based Learning（如 PET、P-tuning）等方法中。
        #
        ## template 是一个列表，表示 prompt 模板中每个 "cloze"（完形填空）部分的长度。
        #  例如：template = [5, 3] 表示有两个填空区域，第一个长 5 个 token，第二个长 3 个 token。
        self.cloze_length = template
        # 构造 mask 列表，标识哪些位置属于 cloze 区域（需要预测或关注的位置）
        # 列表长度为两个 cloze 长度之和；
        # 注意：此处为嵌套列表；外层是 batch 维度（虽然只有一行），内层是序列长度。
        # 如：cloze_length=[2, 3]
        #    cloze_mask = [1, 1, 1, 1, 1]
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
        ]
        # 将 Python 列表转为 PyTorch 张量；
        # .bool(): 将张量值转为布尔类型（True/False），便于后续用作 mask【1 -> 转为 True】
        # 在前向传播中，可以用它来索引或掩码出 cloze 区域的隐藏状态。
        #
        # 如：假设 cloze_length=[2,3]，tensor shape 为 [1, 5]：tensor([[True, True, True, True, True]])
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        # self.cloze_mask[0] 是第一个（也是唯一一个）序列的 mask，长度为 cloze_length[0] + cloze_length[1]
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)

        if args.prompt_encoder_type == "lstm":
            self.prompt_encoder = Encoder_BiLSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size // 2,
                num_layers=2,
                dropout=0.0,
                bidirectional=True,
                batch_first=True
            )
        elif args.prompt_encoder_type == "transformer":
            self.prompt_encoder = Encoder_Transformer(
                d_model=self.hidden_size,
                nhead=8,
                num_layers=6,
                max_len=len(self.cloze_mask[0])
            )
        #### 创建一个可训练的嵌入层，用于生成 soft prompt 的初始向量表示。
        ##   self.embedding 相当于创建了一个可训练的查找表，该表可在 forward 前向过程中供查询使用
        #
        # 总共 len(self.cloze_mask[0]) 个虚拟 token（即整个 prompt 长度）
        # 每个 token 映射为 embed_size 维的向量
        # 这些向量在训练过程中会被优化，相当于“学习一段最优的 prompt”
        #
        ## torch.nn.Embedding(n_embeddings, embedding_dim): 向量个数、向量纬度
        # 1、PyTorch 的嵌入层，用于将离散的索引映射为连续的向量。
        # 2、结构上是一个可训练的二维张量：[vocab_size, embedding_dim]
        #
        ## len(self.cloze_mask[0])
        # cloze_mask 是一个布尔张量，形状通常是 [1, total_length] 或 [total_length]
        # len(cloze_mask[0]) 取出第一个序列（即唯一一个模板序列）, 并计算其长度，
        # 表示要为【多少个位置】创建可训练的嵌入向量。这些“位置”不是词表中的 token，
        # 而是 prompt 模板中的虚拟 token 位置（例如 soft prompt 的每个 slot）
        #
        ## self.embed_size 每个嵌入向量的维度。
        # 通常等于模型的 hidden_size（如 BERT 的 768），以便与模型输入对齐
        #
        ## .to(self.device) 将嵌入层移动到指定设备（CPU/GPU）
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.embed_size).to(self.device)

        #### 创建两层的 MLP（多层感知机），也叫 Projection Head 或 Transformation Head。
        ## MLP 作用：将随机初始化的 prompt embeddings 进行非线性变换，
        # 使其更接近真实 token 的 embedding 分布，以提升训练稳定性和效果
        #
        ## 实验表明，在 P-tuning 中加入 MLP 能显著提升性能。
        #
        ## self.embedding 和 self.mlp_head 都是可训练参数，在训练时会被优化。
        #
        ## nn.Linear(in, out) 全连接层，参数：in_features, out_features 输入、输出维度
        #
        ## 最后一层线性变换：将特征映射回原始维度
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        print("init prompt encoder...")

    def forward(self) -> Tensor:
        """
        对所有 prompt token 嵌入向量进行学习
        1、根据 index 查找 prompt token 的嵌入；
        2、通过 prompt agent 进行编码；
        3、将编码结果经过 MLP 后，得到输出嵌入向量
        Returns:
            学习后的 prompt token 的嵌入向量
        """
        '''
        ## 从 self.embedding 查找表中按 self.seq_indices 索引指定的顺序和位置取出嵌入向量。
           输出是一个张量，形状为[num_embeddings, embedding_dim], 其中 num_embeddings = len(self.seq_indices)
           索引指示取出【哪些位置的向量、顺序怎样、是否重复】。比如：
             self.embedding([0, 0, 1, 1])  # 可以重复使用同一个向量
             self.embedding([3, 2, 1, 0])  # 可以反序
              
        ## unsqueeze() 用来给张量（tensor）增加一个新的维度，维度大小为 1。常用于：
           1、给数据增加 batch 维度（从 (seq_len, dim) 变成 (1, seq_len, dim)）
           2、为了满足某些操作对输入维度数量的要求（比如矩阵乘法、卷积等）
           
        ## embedding(input_ids) 实际上是：embedding.__call__(input_ids) → embedding.forward(input_ids)   
        '''
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.prompt_encoder(input_embeds)
        '''
        squeeze() 删除张量（tensor）中“所有”大小为 1 的维度;squeeze(dim)：只删除指定且大小为 1 的维度。
        常用在：
        1、去掉多余的 batch 维度（当 batch_size = 1 时）
        2、清理计算过程中产生的不必要的单元素维度
        '''
        output_embeds = self.mlp_head(output_embeds).squeeze()
        return output_embeds


class Encoder_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, batch_first):
        super(Encoder_BiLSTM, self).__init__()
        self.lstm_head = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first
        )

    def forward(self, inputs):
        outputs = self.lstm_head(inputs)[0]

        return outputs


class Encoder_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len):
        """

        Args:
            d_model:
            nhead:
            num_layers:
            max_len:
        """
        super(Encoder_Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pos_embedding = PositionalEncoding(d_model, 0.1, max_len)

    def forward(self, inputs):
        """

        Args:
            inputs:

        Returns:

        """
        input_embedding = self.pos_embedding(inputs)
        input_embedding = input_embedding.permute(1, 0, 2)

        outputs = self.encoder(input_embedding).permute([1, 0, 2])

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2501):
        """

        Args:
            d_model:
            dropout:
            max_len:
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(
                10000.0
            ) / d_model)
        )  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = x + self.pe

        return self.dropout(x)


def create_model(args, use_lm_finetune: bool) -> PreTrainedModel:
    """
    PreTrainedModel 继承 torch.nn.Module，是所有 Auto 模型的基类
    根据指定路径（args.model_name_or_path）加载一个因果语言模型（如 GPT），
    如果不需要微调，则将其转为半精度以节省显存；否则保持全精度用于训练。
    :param args:
    :param use_lm_finetune: 是否使用微调

    """
    # AutoModelForCausalLM 自动加载一个自回归语言模型（如 GPT 系列）
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # 如果不进行微调（即 use_lm_finetune 为 False），则将模型的参数转换为半精度浮点数（16位）
    # .half() 是 PyTorch 的方法，将模型从默认的 float32 转换为 float16
    # 目的：减少显存占用，加快推理速度，适用于仅进行推理（inference）而不训练的场景。
    if not use_lm_finetune:
        model = model.half()
    return model


def get_embedding_layer(args, model) -> torch.nn.Embedding:
    """
    从一个预训练语言模型中提取其输入嵌入层（Input Embedding Layer），即，
    将输入的 token ID 映射为【词向量（word embeddings）】的那层。
    :return torch.nn.Embedding(vocab_size, hidden_size)，
            vocab_size：词表大小（如 30522 for BERT）
            hidden_size：嵌入向量维度（如 768 for BERT）
    """
    #### base_model 是 Hugging Face 模型结构中的常见属性;
    # 作用是访问模型的“主干网络”部分，而不是顶层的输出头（如 lm_head）
    #
    # 对于使用 PeftModel（如 LoRA 微调）的模型，model.base_model 指向底层的原始预训练模型;
    # 对于普通模型（如直接用 AutoModel.from_pretrained() 加载的），
    # model.base_model 通常指向模型自身的主干部分（如 RobertaModel、LlamaModel 等）。
    #
    #### .get_input_embeddings() 是 Hugging Face Transformers 模型提供的一个标准方法。
    # 它返回模型的输入嵌入层，即，将输入的 token ID 映射为【词向量（word embeddings）】的那层。
    # 通常是 torch.nn.Embedding 类型的一个模块。
    # 例如，对于 GPT、BERT、LLaMA 等模型，输入是 token IDs，第一层就是把这个 ID 查表转换成一个稠密向量（embedding）。
    return model.base_model.get_input_embeddings()

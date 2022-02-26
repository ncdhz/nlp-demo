from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss

class QAModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None
        ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.qa_outputs(outputs[0])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        question_answer_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            question_answer_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=question_answer_loss,
            start_logits=start_logits,
            end_logits=end_logits
        )
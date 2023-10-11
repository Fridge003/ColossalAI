import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence Roberta
# ===============================


# define data gen function
def data_gen():
    # Generated from following code snippet
    #
    # from transformers import RobertaTokenizer
    # input = 'Hello, my dog is cute'
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # tokenized_input = tokenizer(input, return_tensors='pt')
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    input_ids = torch.tensor([[0, 31414, 6, 127, 2335, 16, 11962, 2]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


# define output transform function
output_transform_fn = lambda x: x

# define loss funciton
loss_fn_for_roberta_model = lambda x: torch.nn.functional.mse_loss(
    x.last_hidden_state, torch.ones_like(x.last_hidden_state)
)
loss_fn = lambda x: x.loss

config = transformers.RobertaConfig(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    hidden_dropout_prob=0,
    attention_probs_dropout_prob=0,
)

# register the Roberta variants
model_zoo.register(
    name="transformers_roberta",
    model_fn=lambda: transformers.RobertaModel(config, add_pooling_layer=False),
    data_gen_fn=data_gen,
    output_transform_fn=output_transform_fn,
    loss_fn=loss_fn_for_roberta_model,
    model_attribute=ModelAttribute(has_control_flow=True),
)

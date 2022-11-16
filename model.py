from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import numpy as np
import torch
from tqdm import tqdm
import random

def get_transformer_model():

	# Feel free to change models if having memory issue
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	model = GPT2LMHeadModel.from_pretrained("gpt2")

	# 'pt' for PyTorch, 'tf' for TensorFlow
	framework = 'pt'

	return TransformerModel(model, tokenizer, framework)


class TransformerModel(object):

	def __init__(self, model, tokenizer, framework='pt'):

		self.model = model
		self.tokenizer = tokenizer
		self.framework = framework

		##### Feel free to add more attributes here if needed #####


	def generate_text(self, prompt, max_new_tokens=10, num_return_sequences=1):
		"""
		The method generates the complementary text for a given starting
		text, i.e., the prompt.

		Args:
			prompt: the starting text as a string
			max_length [optional]: the max length of the generated text

		Return:
			results: the generated text as a string.
		"""

		##### Your code here #####
		# if max_new_tokens == 2:
		# 	return prompt + 'positive'
		
		# results = [prompt + ' with an output placeholder.\n', 
		# 		   prompt + ' with another output placeholder.\n']

		##### Code done #####
		if len(prompt)>1024:
			prompt = prompt[-1024:]
		input_ids = self.tokenizer(prompt, return_tensors = self.framework).input_ids

		if max_new_tokens == 2:
			outputs = self.model.generate(input_ids,
										pad_token_id = self.model.config.eos_token_id,
										do_sample = True,
										no_repeat_ngram_size = 1,
										temperature = 0.3,
										top_k = 1,
										penalty_alpha = 0.6,
										max_new_tokens = max_new_tokens)

			result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
			return result

		outputs = self.model.generate(input_ids,
										do_sample = True,
										pad_token_id = self.model.config.eos_token_id,
										no_repeat_ngram_size = 3,
										num_beams = 5,
										max_new_tokens = max_new_tokens, 
										top_p = 0.90,
										temperature = 0.9,
										num_return_sequences = num_return_sequences)

		results = ""
		for i, sample_output in enumerate(outputs):
			results += "\n\n\n"
			results += "{}: {}".format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True)) 
			
		return results


	def evaluate_ppl(self, dataset):
		"""
		The method for evaluating the perplexity score on given datasets,
		e.g., WikiText-2.

		Args:
			dataset: a `datasets.Dataset' instance from Huggingface

		Return:
			score: A float number. The perplexity score.
		"""

		##### Your code here #####
		tokenizer = self.tokenizer

		test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
		encodings = self.tokenizer("\n\n".join(test["text"]), return_tensors="pt")

		max_length = self.model.config.n_positions
		stride = 512
		seq_len = encodings.input_ids.size(1)

		nlls = []
		prev_end_loc = 0
		for begin_loc in tqdm(range(0, seq_len, stride)):
			end_loc = min(begin_loc + max_length, seq_len)
			trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
			input_ids = encodings.input_ids[:, begin_loc:end_loc]
			target_ids = input_ids.clone()
			target_ids[:, :-trg_len] = -100

			with torch.no_grad():
				outputs = self.model(input_ids, labels=target_ids)

				# loss is calculated using CrossEntropyLoss which averages over input tokens.
				# Multiply it with trg_len to get the summation instead of average.
				# We will take average over all the tokens to get the true average
				# in the last step of this example.
				neg_log_likelihood = outputs.loss * trg_len

			nlls.append(neg_log_likelihood)

			prev_end_loc = end_loc
			if end_loc == seq_len:
				break

		score = torch.exp(torch.stack(nlls).sum() / end_loc)

		##### Code done #####

		return score


	def get_template(self, doc, lbl):
		##### Write your own template below #####
		template = 'Review: \"%s\"\nSentiment: %s' %(doc, lbl)
		##### Template done #####

		return template


	def fewshot_sentiment(self, trainSet, test_doc):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
		Return:
			prediction: String. The predicted sentiment, 'positive' or 
						'negative'.
		"""

		prompt = ''
		for (doc, lbl) in trainSet:
			prompt += self.get_template(doc, lbl)
			prompt += '\n###\n'

		prompt += self.get_template(test_doc, "")

		# 'positive'/'negative' plus an EoS token
		prediction = self.generate_text(prompt, max_new_tokens=2)

		return prediction.split('\n###\n')[-1]


	def visualize_attention(self, trainSet, test_doc, layer=-1):
		"""
		(Bonus) Visualize how attention works in the fewshot sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
			layer: Integer. To speficify which attention layer to be visualized.
		Return:
			template: The template input to the language model.
			weights: 1D-Array. The attention score of each token in the template.
					 Values should be in [0,1], normalize if needed.
		"""

		prompt = ''
		for (doc, lbl) in trainSet:
			prompt += self.get_template(doc, lbl)
			prompt += '\n###\n'

		prompt += self.get_template(test_doc, "")

		##### Your code here #####

		inputs = self.tokenizer.encode(prompt, return_tensors=self.framework)  # Tokenize input text
		outputs = self.model(inputs.cuda())  # Run model
		attention = outputs[layer]  # Retrieve attention from model outputs
		tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])
		inputs_ = self.tokenizer.encode(' ', return_tensors='pt')  # Tokenize input text
		tokens_ = self.tokenizer.convert_ids_to_tokens(inputs_[0])
		#print(tokens_)
		for i in range(len(tokens)):
			if tokens_[0] in tokens[i]:
				tokens[i]=tokens[i].replace(tokens_[0],'')

		inputs_ = self.tokenizer.encode('\n', return_tensors='pt')  # Tokenize input text
		tokens_ = self.tokenizer.convert_ids_to_tokens(inputs_[0])
		#print(tokens_)
		for i in range(len(tokens)):
			if tokens_[0] in tokens[i]:
				tokens[i]=tokens[i].replace(tokens_[0],'\n')
		
		i=3
		weights=np.squeeze(np.mean(attention[1].cpu().detach().numpy(),axis=-1))
		weights=np.mean(weights,axis=0)
		weights=(weights-np.min(weights))/(np.max(weights)-np.min(weights))
		print(weights.shape)
		print(len(tokens))

		##### Code done #####
		assert len(prompt.split())==len(weights)

		return prompt, weights


	def finetune(self, trainSet):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
		"""
		templates = [{"text": self.get_template(doc, lbl)} for doc, lbl in trainSet]
		dataset = Dataset.from_list(templates)
		# Use "left" truncation so that the sentiment is not truncated.
		map_tokenize = lambda x: self.tokenizer(x['text'], truncation_side='left')
		dataset = dataset.map(map_tokenize, batched=True)
		dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)

		##### Your code here #####

		dataset = GPT2Dataset(trainSet, self.tokenizer)

		# Split into training and validation sets
		train_size = int(0.9 * len(dataset))
		val_size = len(dataset) - train_size

		train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

		print('{:>5,} training samples'.format(train_size))
		print('{:>5,} validation samples'.format(val_size))

		train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = 1 # Trains with this batch size.
        )

		#print(next(iter(train_dataloader)))

		validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = 1 # Evaluate with this batch size.
        )

		self.model.resize_token_embeddings(len(self.tokenizer))

		seed_val = 42

		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.cuda.manual_seed_all(seed_val)
		


		epochs = 10
		learning_rate = 1e-4
		warmup_steps = 1e2
		epsilon = 1e-8
		optimizer = torch.optim.AdamW(self.model.parameters(),
								lr = learning_rate,
								eps = epsilon
								)

		total_steps = len(train_dataloader) * epochs

		scheduler = get_linear_schedule_with_warmup(optimizer, 
													num_warmup_steps = warmup_steps, 
													num_training_steps = total_steps)



		training_stats = []

		self.model = self.model

		for epoch_i in range(0, epochs):

			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
			print('Training...')
			
			total_train_loss = 0
			
			self.model.train()
			
			for step, batch in enumerate(train_dataloader):
				print(step)
				b_input_ids = batch[0]
				b_labels = batch[0]
				b_masks = batch[1]
				
				self.model.zero_grad()
				
				outputs = self.model(b_input_ids,
                          		labels = b_labels, 
                          		attention_mask = b_masks,
                          		token_type_ids = None
				)
				loss = outputs[0]
				
				batch_loss = loss.item()
				
				total_train_loss += batch_loss
					
				loss.backward()
				
				optimizer.step()
				
				scheduler.step()
				
			# Calculate the average loss over all of the batches.
			
			avg_train_loss = total_train_loss / len(train_dataloader)
			
			print("")
			
			print("  Average training loss: {0:.2f}".format(avg_train_loss))

			print("")
			print("Running Validation...")
    
			self.model.eval()
			
			total_eval_loss = 0
			
			nb_eval_steps = 0

    		# Evaluate data for one epoch
			
			for batch in validation_dataloader:
				
				b_input_ids = batch[0]
				b_labels = batch[0]
				b_masks = batch[1]
				
				with torch.no_grad():
					outputs = self.model(b_input_ids,
										attention_mask = b_masks,
										labels = b_labels
										)
										
					loss = outputs[0]
				
				batch_loss = loss.item()
				
				total_eval_loss += batch_loss
				
			avg_val_loss = total_eval_loss / len(validation_dataloader)
			
			print("  Validation Loss: {0:.2f}".format(avg_val_loss))


		self.model

		##### Code done #####
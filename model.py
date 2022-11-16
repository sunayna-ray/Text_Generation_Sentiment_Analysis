from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
import numpy as np

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
		input_ids = self.tokenizer(prompt, return_tensors = self.framework).input_ids

		if max_new_tokens == 2:
			outputs = self.model.generate(input_ids,
										pad_token_id = self.model.config.eos_token_id,
										#do_sample = True,
										no_repeat_ngram_size = 1,
										#top_k = 1,
										#penalty_alpha = 0.6,
										temperature = 0.3,
										max_new_tokens = max_new_tokens)

			result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
			return result

		outputs = self.model.generate(input_ids,
										do_sample = True,
										pad_token_id = self.model.config.eos_token_id,
										no_repeat_ngram_size = 3,
										top_p = 0.90,
										temperature = 0.9,
										num_beams = 5,
										max_new_tokens = max_new_tokens, 
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
		
		score = 100.0

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

		weights = np.random.rand(len(prompt.split()))

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



		##### Code done #####
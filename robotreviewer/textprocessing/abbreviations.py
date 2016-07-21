
test_sentences = ["Long-term androgen suppression plus radiotherapy (AS+RT) is standard treatment of high-risk prostate cancer.",
	"To compare the test-retest reliability, convergent validity, and overall feasibility/ usability of activity-based (AB) and time-based (TB) approaches for obtaining self-reported moderate-to-vigorous physical activity (MVPA) from adolescents.",
	"This study was conducted to determine if prophylactic cranial irradiation (PCI) improves survival in locally advanced non-small-cell lung cancer (LA-NSCLC)",
	"Alternatives to cytotoxic agents are desirable for patients with HIV-associated Kaposi's sarcoma (KS).",
	"The primary objective was assessment of antitumor activity using modified AIDS Clinical Trial Group (ACTG) criteria for HIV-KS.",
	"To determine the effectiveness of bortezomib plus irinotecan and bortezomib alone in patients with advanced gastroesophageal junction (GEJ) and gastric adenocarcinoma."]

import re

class Abbreviations:

	def __init__(self, source_text):
		self.dictionary = self.make_dictionary(source_text)

	def sub(self, text):
		for k, v in self.dictionary.items():
			text = re.sub(r"(\W)"+k+r"(\W)", r"\1"+v+r"\2", text)
		return text


	def make_dictionary(self, text):

		s_l = text.lower()

		# first get indices of abbreviations
		index_groups = [(m.start(0), m.end(0)) for m in re.finditer('\([a-z\+\-]+\)', s_l)]

		lookup = {}

		for start_i, end_i in index_groups:
			
			abbreviation = re.sub('[^a-z]', '', s_l[start_i+1: end_i-1].lower())

			if abbreviation in lookup or not abbreviation:
				# skip if we already know (likely to be defined
				# the first time only so more efficient);
				# also skip if empty abbreviation (i.e. non-text
				# characters in brackets)
				continue

			abbreviation_i = len(abbreviation)-1

			
			end_j = start_i-1
			span = None

			for i in range(end_j, min(0, end_j-100), -1):

				

				if abbreviation_i == 0:
					# the first character of the abbreviation has to be the start
					# of a word (the others not necessarily)
					abbreviation_char = " " + abbreviation[abbreviation_i]
					text_char = s_l[i-1:i+1]
				else:
					abbreviation_char = abbreviation[abbreviation_i]
					text_char = s_l[i]

				

				if abbreviation_char == text_char:
					abbreviation_i -= 1
					if abbreviation_i == -1:
						span = i, end_j
						break
			
			if span is not None:
				lookup[text[start_i+1:end_i-1]] = text[span[0]:span[1]]

		return lookup



def main():
	for s in test_sentences:

		d = Abbreviations(s)
		print(d.dictionary)
		print(s)
		print(d.sub(s))


if __name__ == '__main__':
	main()







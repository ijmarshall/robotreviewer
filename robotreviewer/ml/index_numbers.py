# three million, two hundred and fourteen thousand, one hundred and twelve

# 3276191

import collections
import re
import timeit #for testing


# TODO
# improve handling of the word 'a'
# ideally will operate as number 1 in front of hundred, thousand etc.
# but not flag as a number otherwise

class Indexer():
    """
    base class for various text taggers

    takes in text; main data structure is a list of tuples
           [tag, start, end]

    where: tag is any data type
           start and end are integers representing the start and end indices in the string
    """

    def __init__(self):
        pass

    def tag(self, text):
        pass


class WordTagger(Indexer):
    """
    simple regular expression word tokenizer
    """

    def tag(self, text):
        self.tags = self.get_words(text)

    def get_words(self, text):
        return [(m.group(), m.start(), m.end()) for m in re.finditer("([\.\,\;']|[a-z0-9]+)", text, re.IGNORECASE) if m.group() not in ['and', ',']]


class NumberTagger(WordTagger):

    def __init__(self):
        self.load_numberwords()
        Indexer.__init__(self)

    def load_numberwords(self):
        self.numberwords = {
            # 'a':      1,
            'one':      1,
            'two':      2,
            'three':    3,
            'four':     4,
            'five':     5,
            'six':      6,
            'seven':    7,
            'eight':    8,
            'nine':     9,
            'ten':      10,
            'eleven':   11,
            'twelve':   12,
            'thirteen': 13,
            'fourteen': 14,
            'fifteen':  15,
            'sixteen':  16,
            'seventeen':17,
            'eighteen': 18,
            'nineteen': 19,
            'twenty':   20,
            'thirty':   30,
            'forty':    40,
            'fifty':    50,
            'sixty':    60,
            'seventy':  70,
            'eighty':   80,
            'ninety':   90,
            'hundred':  100,
            'thousand': 1000,
            'million': 1000000,
            'billion': 1000000000,
            'trillion': 1000000000000
        }

    def swap(self, text):
        """
        returns string with number words replaced with digits
        """
        text = re.sub(r"(?<=[0-9])[\s\,](?=[0-9])", "", text)
        tags = self.tag(text)
        # tags.sort(key=lambda (number, start, end): start) # get tags and sort by start index
        tags.sort(key=lambda indices: indices[1])

        output_list = []
        progress_index = 0

        for (number, start_index, end_index) in tags:
            output_list.append(text[progress_index:start_index]) # add the unedited string from the last marker up to the number
            output_list.append(str(number)) # add the string digits of the number
            progress_index = end_index # skip the marker forward to the end of the original number words

        output_list.append(text[progress_index:]) # if no tags, this will append the whole unchanged string

        return ''.join(output_list)


    def tag(self, text):
        """
        produces a list of tuples (number, start_index, end_index)
        """
        words = self.get_words(text)
        words.reverse()

        number_parts = []
        number_parts_index = -1

        last_word_was_a_number = False

        # first get groups of consecutive numbers from the reversed word list



        for word, start, end in words:

            word_num = self.numberwords.get(word.lower())

            if word_num is None:
                last_word_was_a_number = False
            else:
                if last_word_was_a_number == False:
                    number_parts.append([])
                    number_parts_index += 1
                last_word_was_a_number = True

                number_parts[number_parts_index].append((word_num, start, end))

        output = []


        # then calculate the number for each part

        for number_part in number_parts:
            number = self.recursive_nums([word_num for word_num, start, end in number_part])
            start = min([start for word_num, start, end in number_part])
            end = max([end for word_num, start, end in number_part])

            output.append((number, start, end))
        return(output)

    def recursive_nums(self, numlist):

        # first split list up

        tens_index = 0
        tens = [100, 1000, 1000000, 1000000000, 1000000000000]

        current_multiplier = 1

        split_list = collections.defaultdict(list)

        for num in numlist:
            if num in tens[tens_index:]:
                tens_index = tens.index(num)+1
                current_multiplier = num
            else:
                split_list[current_multiplier].append(num)

        counter = 0

        # then sum up the component parts

        for multiplier, numbers in split_list.items():
            # check if multiples of ten left

            for number in numbers:
                if number in tens:
                    counter += multiplier * self.recursive_nums(numbers)
                    break
            else:
                counter += multiplier * sum(numbers)

        return counter



    # counter = 0

    # for i, num in enumerate(numlist):
    #   if num % 10 == 0:
    #       counter += (num * recursive_nums(numlist[i+1:]))
    #   else:
    #       counter += num

    # return counter

_swap_num = NumberTagger().swap
def swap_num(text):
    return _swap_num(text)



def test(t):
    b = t.tag("""Specific immunotherapy is still widely used in grass-pollen allergy,
 but its side effects may limit its use. We tested the safety and efficacy of a
 formalinized high-molecular-weight allergoid prepared from a mixed grass-pollen
extract with two injection schedules in a double-blind, placebo-controlled study.
 Eighteen patients received placebo, 19 received the low-dose schedule (maximal
dose: 2000 PNU) and 20 received the high-dose schedule (maximal dose: 10,000 PNU).
 Only one patient presented a systemic reaction of moderate severity for a dose
 of 1200 PNU. Before the onset of the pollen season, patients had a nasal challenge
 with orchard grass-pollen grains, a skin test titration, and the titration
of serum-specific IgG. Both groups of patients presented a significant reduction in nasal
and skin sensitivities and a significant increase in IgG compared to placebo. Symptoms and
medications for rhinitis and asthma were studied during the season, and both groups receiving
allergoids had a significant reduction of symptom-medication scores for nasal and bronchial
symptoms. There was a highly significant correlation between nasal symptom-medication
scores during the season and the results of nasal challenges. High-molecular-weight
allergoids are safe and effective.""")




def main():

    t = NumberTagger()

    # b = t.swap("""Specific immunotherapy is still widely used in grass-pollen allergy, but its side effects may limit its use. We tested the safety and efficacy of a formalinized high-molecular-weight allergoid prepared from a mixed grass-pollen extract with two injection schedules in a double-blind, placebo-controlled study. Eighteen patients received placebo, 19 received the low-dose schedule (maximal dose: 2000 PNU) and 20 received the high-dose schedule (maximal dose: 10,000 PNU). Only one patient presented a systemic reaction of moderate severity for a dose of 1200 PNU. Before the onset of the pollen season, patients had a nasal challenge with orchard grass-pollen grains, a skin test titration, and the titration of serum-specific IgG. Both groups of patients presented a significant reduction in nasal and skin sensitivities and a significant increase in IgG compared to placebo. Symptoms and medications for rhinitis and asthma were studied during the season, and both groups receiving allergoids had a significant reduction of symptom-medication scores for nasal and bronchial symptoms. There was a highly significant correlation between nasal symptom-medication scores during the season and the results of nasal challenge High-molecular-weight allergoids are safe and effective.""")
    b = t.swap('three million, two hundred and fourteen thousand, one hundred and twelve')
    print (b)



    # three million, two hundred and fourteen thousand, one hundred and twelve
    # testnums = [12, 100, 1, 1000, 14, 100, 2, 1000000, 3]
    # testnums = [100, 2, 1000000, 40]
    # output 3214112
    # testnums = [1, 90, 100, 1, 1000, 6 ,70, 100, 2, 1000000, 3]

    # testnums = [6, 70, 100, 2]

    # testnums = [1, 90]
    # testanswer = 3276191

    # print recursive_nums(testnums)


if __name__ == '__main__':
    main()

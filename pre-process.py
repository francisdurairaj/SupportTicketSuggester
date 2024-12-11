# pip install numpy requests nlpaug

import csv
import nlpaug.augmenter.word as naw

csvfile="input.csv"
outputcsvfile="output100.csv"

def getSynonymAug(input):
  augmented_text = []
  aug = naw.SynonymAug(aug_src='wordnet',aug_max=100)
  for _ in range(100):
    augmented_text.append(aug.augment(input))
  final= '. '.join([text[0] for text in augmented_text])
  return final

# read csv and augement the description
new_output=[]
with open(csvfile, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      process_data= {
          "category": row[0],
          "sub_category": row[1],
          "description": row[2],
          "url": row[3],
          "detail_description":getSynonymAug(row[2])
      }
      new_output.append(process_data)
      # print(process_data)

# created the output file
with open(outputcsvfile, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in new_output:      
      spamwriter.writerow([row['category'],row['sub_category'],row['description'],row['url'],row['detail_description']])

print("Sucessfully processed input file!!")

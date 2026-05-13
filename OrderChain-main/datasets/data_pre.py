import pandas as pd
import json
import os
df = pd.read_csv(r"/home/duomeitinrfx/data/Aidence/Adience_test.csv")

json_data = []
human_1 = "<image>\nYou are a Junior Facial Analyst and you need to estimate the age group of a person based on the facial features of the person in the provided facial image. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Please first coarsely categorise the images: young (stage0-stage3) or old (stage4-stage7) and output the coarsely categorised results as #<answer>#."
for idx, row in df.iterrows():
    if row['label'] == 0:
        gpt_1 = "#young#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Baby (stage0-stage1) or Teenager (stage2-stage3) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Baby&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Baby (stage0-stage1)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$0$"
    elif row['label'] == 1:
        gpt_1 = "#young#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Baby (stage0-stage1) or Teenager (stage2-stage3) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Baby&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Baby (stage0-stage1)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$1$"
    elif row['label'] == 2:
        gpt_1 = "#young#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Baby (stage0-stage1) or Teenager (stage2-stage3) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Teenager&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Teenager (stage2-stage3)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$2$"
    elif row['label'] == 3:
        gpt_1 = "#young#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Baby (stage0-stage1) or Teenager (stage2-stage3) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Teenager&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Teenager (stage2-stage3)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$3$"
    elif row['label'] == 4:
        gpt_1 = "#old#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Adult (stage4-stage5) or Elder (stage6-stage7) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Adult&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Adult (stage4-stage5)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$4$"
    elif row['label'] == 5:
        gpt_1 = "#old#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Adult (stage4-stage5) or Elder (stage6-stage7) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Adult&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Adult (stage4-stage5)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$5$"
    elif row['label'] == 6:
        gpt_1 = "#old#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Adult (stage4-stage5) or Elder (stage6-stage7) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Elder&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Elder (stage6-stage7)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$6$"
    elif row['label'] == 7:
        gpt_1 = "#old#"
        human_2 = "<image>\nYou are now an Intermediate Facial Analysis Specialist and you need to estimate the age groups of the people in the provided facial images based on their facial features. These are all the candidate age groups: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). Previously, the correct answer for the coarse classification was 'young' (stage0-stage3), you need to further classify the given image as Adult (stage4-stage5) or Elder (stage6-stage7) based on the coarse classification judgment and output the fine classification as &<answer>&."
        gpt_2 = "&Elder&"
        human_3 = "<image>\nYou are now an advanced facial analyst, and you need to estimate the age group of the person in the provided facial image based on their facial features. The candidate age groups are as follows: stage0 (0-2 years old), stage1 (4-6 years old), stage2 (8-13 years old), stage3 (15-20 years old), stage4 (25-32 years old), stage5 (38-43 years old), stage6 (48-53 years old), and stage7 (60+ years old). The previously fine-grained classification result is 'Elder (stage6-stage7)'. Based on this fine classification, you need to make the final age prediction for the given facial image. Please output the corresponding age group number directly in the format $<answer>$."
        gpt_3 = "$7$"
    conversation_1 = [{
                "from": "human",
                "value": human_1
            },
            {
                "from": "gpt",
                "value": gpt_1
            }]
    conversation_2 = [{
                "from": "human",
                "value": human_2
            },
            {
                "from": "gpt",
                "value": gpt_2
            }]
    conversation_3 = [{
                "from": "human",
                "value": human_3
            },
            {
                "from": "gpt",
                "value": gpt_3
            }]

    for i in range(3):
        if i==0:
            conversation = conversation_1
        elif i==1:
            conversation = conversation_2
        elif i==2:
            conversation = conversation_3
        entry = {
            "id": str(idx*3+i),
            "image": row['image_path'], 
            "conversations": conversation
        }
        json_data.append(entry)

output_json = r"/home/duomeitinrfx/data/Aidence/Adience_llava_test.json"
output_dir = os.path.dirname(output_json)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_json, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"successfully!")
library(dplyr)
library(tidyr)
library(jsonlite)
library(stringr)
library(readxl)
####################################################################
# Setup
####################################################################
rm(list = ls())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
parent_path <- getwd()
getwd()
# -------------------------------------
#Seed for reproducibility
# -------------------------------------
set.seed(1848)


####################################################################
# Load Training Data from csv-files
####################################################################

df_raw <- readxl::read_xlsx("../data/framing/annotation_sample_for_ras_annotated_combined.xlsx")
df_raw <- df_raw %>% mutate(frame_present_ra_2 = ifelse(frame_present_ra_2 == "no", 0,
                                                        ifelse(frame_present_ra_2 == "yes", 1, NA)),
                            frame_label_ra_1 = tolower(frame_label_ra_1),
                            frame_label_ra_2 = tolower(frame_label_ra_2)) %>%
                     mutate(check_rel = ifelse(frame_present_ra_2 == frame_present_ra_1, T, F),
                            check_fra = ifelse(frame_label_ra_2 == frame_label_ra_1, T, F))

df <- df_raw %>% filter(check_fra == T)

df <- df %>% mutate(answer = frame_label_ra_1,
                    input = "",
                    output = "",
                    instruction = "")
####################################################################
# Transform data to json for MLX LLM Fine Tuning
####################################################################
json_df <- df %>% select(c(instruction,input,output,answer))


# output generation:
output <- paste0("the correct answer is ", df$answer)

# instruction:
instruction <- paste0("Read the sentence or paragraph provided and dtermine which of the following thematic frame fits best for the sentence or paragraph. According to Robert Entman's definition where framing suggests that frames select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described. Determine if the frame could be classified as: AI Limitations (1) [This frame discusses the shortcomings, challenges, and areas where AI technologies are insufficient or problematic. Understanding the limitations of AI is crucial for developing realistic expectations and identifying areas needing improvement.]; AI Benefits (2) [This frame highlights AI's current positive impacts, including efficiencies, improvements in various fields, and enhancements in quality of life. It helps illustrate the tangible advantages brought by AI technologies.]; AI Risks (3) [This frame emphasizes tangible dangers, potential harms, and negative consequences of AI. Addressing safety, security, and potential misuse of AI systems is critical.]; AI Ethics (4) [This frame addresses moral principles, ethical dilemmas, and the responsibilities of AI technology developers and users. It ensures that discussions consider the moral implications of AI deployment.]; AI Potential (5) [This frame emphasizes future possibilities, advancements, and innovations that AI could bring. It is key to understanding AI's visionary aspects and its role in future developments.]; AI Innovation (6) [This frame highlights specific new ideas, breakthroughs, and novel applications of AI technologies. It showcases the cutting-edge aspects of AI research and development.]; AI Development (7) [This frame discusses the progress, stages of development, and growth trajectory of AI technologies. It provides insights into how AI is evolving.]; AI Impact (8) [This frame describes AI's effects, changes, and influence on various sectors, including society, the economy, and everyday life. It helps to gauge the broad implications of AI technologies.]; AI Regulation (9) [This frame discusses the need to establish and enforce AI rules and policies. It is essential to understand the legal and regulatory landscape shaping AI use.]; AI Concerns (10) [This frame expresses general worries, distress, and issues stakeholders may have about AI. It encapsulates a broad spectrum of anxieties related to AI development and deployment.]; NO FRAME (11) [This category captures all sentences that fail to address any identified frames related to artificial intelligence, such as benefits, risks, regulations, ethics, etc.] Here is the Sentence/Paragraph: ", df$sentences)


json_df$instruction <- instruction
json_df$output <- output


json_df_t <- json_df[1:150,]
json_df_tt <- json_df[151:nrow(json_df),]

write_json(json_df_t, "../data/framing/train.json")
write_json(json_df_tt, "../data/framing/test.json")

                   #FINAL EXAM
                                     # MD SUHEB
                                     # 19MBMB18
                                     # MBA BA
# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
install.packages("syuzhet") # for sentiment analysis
install.packages("ggplot2") # for plotting graphs
install.packages("ROAuth")
install.packages("twitteR")
install.packages("RCurl")
install.packages('devtools')
# Load packages
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library("syuzhet")
library("ggplot2")
library(ROAuth)
library(twitteR)
library(RCurl)
library(devtools)
library(tm)

#Connecting with Twitter API
setup_twitter_oauth('ZwmCJG6yELkeKZ3Frtzd7a8l3', 
                    'kDMtbEqtWTfwPfp1m2fHwMBcIRPzBwv6HyVd5tzGdHwIuBdiPR',
                    access_token='291647217-tYTU5OxH1AvHDGUYC7vvaFQ2COx0I2J41gKMMVml', 
                    access_secret='U6OZJnXKcXG4yM8uaYMMG7uk7ViWm71Q2JbEuMTY5OnRb')

#Loading the tweets
Vijay.tweets=searchTwitter("@VIJAYRuledTwitter2020",n=1500)
Vijay.tweets

#Analyzing the collected data for length,class etc
head(Vijay.tweets)
length(Vijay.tweets)
class(Vijay.tweets)
class(Vijay.tweets[[1]])
Vijay.tweets[[1]]
selectedTweet=Vijay.tweets[[3]]
user<-getUser("VidhuBhushan")
user$created
user$location
user$statusesCount
user$followersCount
user$lastStatus
user$friendsCount

#Converting the data to data frame and storing as a csv file
library(plyr)
tweets.df=ldply(Vijay.tweets,function(t) t$toDataFrame())
summary(tweets.df)
setwd("C:/Users/suhebm/Desktop/Text analytics")
getwd()
write.csv(tweets.df, "vijaytweets.csv")
#Collecting tweets with the name vijay
some_tweets = searchTwitter("vijay", n=1500, lang="en")
#collecting the text
TextDoc = sapply(some_tweets, function(x) x$getText())
#choose file
text <- readLines(file.choose())


# Load the data as a corpus
TextDoc <- Corpus(VectorSource(text))
#Replacing "/", "@" and "|" with space
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
TextDoc <- tm_map(TextDoc, toSpace, "/")
TextDoc <- tm_map(TextDoc, toSpace, "@")
TextDoc <- tm_map(TextDoc, toSpace, "\\|")
# Convert the text to lower case
TextDoc <- tm_map(TextDoc, content_transformer(tolower))
# Remove numbers
TextDoc <- tm_map(TextDoc, removeNumbers)
# Remove english common stopwords
TextDoc <- tm_map(TextDoc, removeWords, stopwords("english"))
# Remove your own stop word
# specify your custom stopwords as a character vector
TextDoc <- tm_map(TextDoc, removeWords, c("s", "company", "team")) 
# Remove punctuations
TextDoc <- tm_map(TextDoc, removePunctuation)
# Eliminate extra white spaces
TextDoc <- tm_map(TextDoc, stripWhitespace)
# Text stemming - which reduces words to their root form
TextDoc <- tm_map(TextDoc, stemDocument)
# Build a term-document matrix
TextDoc_dtm <- TermDocumentMatrix(TextDoc)
dtm_m <- as.matrix(TextDoc_dtm)
# Sort by descearing value of frequency
dtm_v <- sort(rowSums(dtm_m),decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v),freq=dtm_v)
# Display the top 5 most frequent words
head(dtm_d, 5)

# Plot the most frequent words
barplot(dtm_d[1:5,]$freq, las = 2, names.arg = dtm_d[1:5,]$word,
        col ="lightgreen", main ="Top 5 most frequent words",
        ylab = "Word frequencies")
#generate word cloud
set.seed(1500)
wordcloud(words = dtm_d$word, freq = dtm_d$freq, min.freq = 0.5,
          max.words=100, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
# Find associations 
findAssocs(TextDoc_dtm, terms = c("good","work","health"), corlimit = 0.25)			
# Find associations for words that occur at least 50 times
findAssocs(TextDoc_dtm, terms = findFreqTerms(TextDoc_dtm, lowfreq = 50), corlimit = 0.25)

# regular sentiment score using get_sentiment() function and method of your choice
# please note that different methods may have different scales
syuzhet_vector <- get_sentiment(text, method="syuzhet")
# see the first row of the vector
head(syuzhet_vector)
# see summary statistics of the vector
summary(syuzhet_vector)
# bing
bing_vector <- get_sentiment(text, method="bing")
head(bing_vector)
summary(bing_vector)
#affin
afinn_vector <- get_sentiment(text, method="afinn")
head(afinn_vector)
summary(afinn_vector)
#compare the first row of each vector using sign function
rbind(
  sign(head(syuzhet_vector)),
  sign(head(bing_vector)),
  sign(head(afinn_vector))
)
# run nrc sentiment analysis to return data frame with each row classified as one of the following
# emotions, rather than a score: 
# anger, anticipation, disgust, fear, joy, sadness, surprise, trust 
# It also counts the number of positive and negative emotions found in each row
d<-get_nrc_sentiment(text)
# head(d,10) - to see top 10 lines of the get_nrc_sentiment dataframe
head (d,10)
#transpose
td<-data.frame(t(d))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td[2:253]))
#Transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2<-td_new[1:8,]
#Plot One - count of words associated with each sentiment
quickplot(sentiment, data=td_new2, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Survey sentiments")
#Plot two - count of words associated with each sentiment, expressed as a percentage
barplot(
  sort(colSums(prop.table(d[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Text", xlab="Percentage"
)

#TOPIC MODELLING
install.packages("topicmodels")
library(topicmodels)
vijaytweets=readLines(file.choose())
#apply topic modeling with k=2
ap_lda <- LDA(dtm_m, k = 2, control = list(seed = 1234))
ap_lda

#1. Word-Topic Probabilities
#The tidytext package provides this method for extracting the per-topic-per-word probabilities, called ??
#("beta")
library(tidytext)
ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics
#Notice that this has turned the model 
#into a one-topic-per-term-per-row format

library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
#This visualization lets us understand the two topics 
#that were extracted from the dataset

#predicting the probabilities
library(tidyr)
beta_spread <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))
head(beta_spread, 15)
beta_spread
#The words with the greatest differences
#between the two topics are visualized
ap_documents <- tidy(ap_lda, matrix = "gamma")
head(ap_documents, 15)
ap_documents
#Each of these values is an estimated proportion of words 
#from that dataset that are generated from that topic
tidy(vijaytweets) %>%
  filter(document == 6) %>%
  arrange(desc(count))

# classify polarity
class_pol = get_sentiment(ap_topics$term)
class_pol
# get polarity best fit
polarity = class_pol

test <- function(polarity) {
  if(polarity < 0) {
    "negative"
  } else if (polarity >0){
    "positive"
  } else {"neutral"}
}

polarity = sapply(polarity, test)
polarity= data.frame(polarity)

ggplot(polarity, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="RdGy") +
  labs(x="polarity categories", y="number of tweets") +
  labs(title = "Sentiment Analysis of Tweets about vijay with topic model k=2 \n(classification by polarity)")

#FOR K=6
docs <- Corpus(VectorSource(some_txt))
dtm <- DocumentTermMatrix(docs)

#apply topic modeling with k=6
ap_lda <- LDA(dtm_m, k = 6, control = list(seed = 1234))
ap_lda

#1. Word-Topic Probabilities

library(tidytext)
ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics

library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

#predicting the probabilities
library(tidyr)
beta_spread <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))
head(beta_spread, 15)

ap_documents <- tidy(ap_lda, matrix = "gamma")
head(ap_documents, 15)
ap_documents

tidy(vijaytweets) %>%
  filter(document == 6) %>%
  arrange(desc(count))


# classify polarity
class_pol = get_sentiment(ap_topics$term)
class_pol
# get polarity best fit
polarity = class_pol

test <- function(polarity) {
  if(polarity < 0) {
    "negative"
  } else if (polarity >0){
    "positive"
  } else {"neutral"}
}

polarity = sapply(polarity, test)
polarity= data.frame(polarity)

ggplot(polarity, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="RdGy") +
  labs(x="polarity categories", y="number of tweets") +
  labs(title = "Sentiment Analysis of Tweets about vijay with topic model k=6 \n(classification by polarity)")

#FOR K=10
docs <- Corpus(VectorSource(some_txt))
dtm <- DocumentTermMatrix(docs)

#apply topic modeling with k=10
ap_lda <- LDA(dtm_m, k = 10, control = list(seed = 1234))
ap_lda

#1. Word-Topic Probabilities

library(tidytext)
ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics

library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

#predicting the probabilities
library(tidyr)
beta_spread <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))
head(beta_spread, 15)

ap_documents <- tidy(ap_lda, matrix = "gamma")
head(ap_documents, 15)
ap_documents
tidy(vijaytweets) %>%
  filter(document == 10) %>%
  arrange(desc(count))


# classify polarity
class_pol = get_sentiment(ap_topics$term)
class_pol
# get polarity best fit
polarity = class_pol

test <- function(polarity) {
  if(polarity < 0) {
    "negative"
  } else if (polarity >0){
    "positive"
  } else {"neutral"}
}

polarity = sapply(polarity, test)
polarity= data.frame(polarity)

ggplot(polarity, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="RdGy") +
  labs(x="polarity categories", y="number of tweets") +
  labs(title = "Sentiment Analysis of Tweets about vijay with topic model k=10 \n(classification by polarity)")
#4)	Pick one model for further analysis, out of the above three models, and give your justification 
#Model with K=10 is good to be condisered for further analysis since Almost any topic model in practice will use a larger k
#5)Out of the three models highest probability if for the terms kollywood 9.92e-1 and fabas 9.92e-1
#6)Yes highest probability is for the same terms in all the three models
#7),8) Graphs plotted
#9)There is some evidence that the words associated with some of the topics cohere into something that could be called a latent theme or topic within the text. For example, topic #1 includes words that are often used to discuss movies and actors and topic #9 appears to describe people opinions. But there are many other topics that don't make much sense.
#10)Topic modeling provides us with methods to organize, understand and summarize large collections of textual information. It helps in: Discovering hidden topical patterns that are present across the collection
#Topic modeling helped in drawing conclusions for which terms in the choosen topic have
#highest probability and out of the 3 models choosen which is the model that can be considered as optical
#It further helped in drawing conclusions for the most common words among the data choosen
#It helped in detecting the pattern of the words and the words with highest repetition
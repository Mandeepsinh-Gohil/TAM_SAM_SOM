import re
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.chrome.options import Options
from langchain_community.llms import Ollama
import docx

import os
from docx import Document
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
nltk.download("vader_lexicon")

from transformers import pipeline
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

from transformers import BartTokenizer, BartForConditionalGeneration



nltk.download('stopwords')
nltk.download('punkt')

# def perplexity_search(query):
#     driver = webdriver.Chrome()
#     driver.minimize_window()
    
#     driver.get("https://www.perplexity.ai/")
#     search_box = driver.find_element(By.TAG_NAME, "textarea")
#     search_box.send_keys(query)
#     search_box.send_keys(Keys.ENTER)
#     time.sleep(15)
    
#     try:
#         result_div = WebDriverWait(driver, 60).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, "div.break-words"))
#         )
#         return result_div.text
#     except Exception as e:
#         print("Error:", e)
#     finally:
#         driver.quit()


def phind_search(query):
  
    driver = webdriver.Chrome()
    driver.minimize_window() 
    
    # query_row = []
    # sse = "Phind"
    # query_row.append(sse)
    driver.get("https://www.phind.com/search?home=true")
    
    search_box = driver.find_element(By.TAG_NAME, "textarea")
    search_box.send_keys(query)
    search_box.send_keys(Keys.ENTER)
    time.sleep(10)
    
    try:
        result_div = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.NAME, "answer-0"))
        )
        final_result = ""
        results = result_div.find_elements(By.CSS_SELECTOR, 'div.fs-5')
        for result in results:
            final_result = final_result + result.text

        return (final_result)
        
        
        # links = []
        # result_links = result_div.find_elements(By.TAG_NAME, "a")
        # for link in result_links:
        #     href = link.get_attribute("href")
        #     links.append(href)
        
        # query_row.append(final_result)
        # query_row.append(links)
        
        # with open("SSE_QueryResults.csv", "a+", newline="") as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(query_row)
        
        print(sse + ": Successfully found and stored results")

    except Exception as e:
        print("Error:", e)

    finally:
        driver.quit()
        
    
def perplexity_search(query):

    driver = webdriver.Chrome()
    driver.minimize_window()
    
    # query_row = []
    # sse = "Perplexity"
    # query_row.append(sse)
    driver.get("https://www.perplexity.ai/")
    
    search_box = driver.find_element(By.TAG_NAME, "textarea")
    search_box.send_keys(query)
    search_box.send_keys(Keys.ENTER)
    time.sleep(15)
    
    try:
        result_div = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.break-words"))
        )
        return (result_div.text)
    except Exception as e:
        print("Error:", e)

    finally:
        driver.quit() 

def llama3_search(query):
    # query_row = []
    # query_row.append("Llama3")
    
    llm = Ollama(model='llama3')
    result = llm.invoke(query)
    return (result)
    
def llama2_search(query):
    # query_row = []
    # query_row.append("Llama2")
    
    llm = Ollama(model='llama2')
    result = llm.invoke(query)
    return (result)  

def mistral_search(query):
    # query_row = []
    # query_row.append("Mistral")
    
    llm = Ollama(model='mistral')
    result = llm.invoke(query)
    return (result)

def extract_and_normalize_values(text):
    
    # tam_pattern = re.compile(r'TAM:\$?(\d+\.?\d+):? (trillion|billion|million)?',re.IGNORECASE)
    # potential_customers_pattern = re.compile(r'(TotalNumberOfPotentialCustomers|Total Number Of Potential Customers):(\d+\.?\d+):?(million|billion|trillion)?',re.IGNORECASE)
    # revenue_per_customer_pattern = re.compile(r'(AverageAnnualRevenuePerCustomer|Average Annual Revenue Per Customer):\$?(\d+\.?\d+):?(billion|million)?', re.IGNORECASE)
    
    # tam = extract_value(tam_pattern, text)
    # potential_customers = extract_value(potential_customers_pattern, text)
    # revenue_per_customer = extract_value(revenue_per_customer_pattern, text)

    

    sam, som = calculate_sam_som(potential_customers, revenue_per_customer)
    return tam, potential_customers, revenue_per_customer, sam, som

def extract_value(pattern, text):
    match = pattern.search(text)
    print("inside extract function : ")
    print("match: ",match)
    print(pattern)
    print(text)
    
    
    if match:
        value, unit = match.groups()
        print("value: ",value,"Unit: ",unit)
        return convert_to_millions(value, unit)
    return None

def convert_to_millions(value, unit):
    value = float(value.replace(',', ''))
    if unit:
        unit = unit.lower()
        if unit == 'billion':
            return value * 1000 
        elif unit.lower() == 'trillion':
            return value * 1000000
        elif unit == 'million':
            return value
    print(value/1000000)
    return value / 1000000 if value > 1000 else value  # Convert raw large numbers to millions if appropriate

def calculate_sam_som(total_customers, revenue_per_customer, market_share=0.1):
    total_customers = float(total_customers) if total_customers else 0
    revenue_per_customer = float(revenue_per_customer) if revenue_per_customer else 0
    
    number_of_target_customers = 0.5 * total_customers
    sam = number_of_target_customers * revenue_per_customer
    som = sam * market_share
    return sam, som

# Define column headers

# phind:"Result_phind", "TAM (in million USD)_phind", "Total Number of Potential Customers (in million)_phind", "Average Annual Revenue Per Customer (in million USD)_phind","SAM (in million USD)_phind",
#     "SOM (in million USD)_phind"
headers = ["Query","Result_perplexity", "TAM (in million USD)_perplexity", "Total Number of Potential Customers (in million)_perplexity", "Average Annual Revenue Per Customer (in million USD)_perplexity","SAM (in million USD)_perplexity",
    "SOM (in million USD)_perplexity","Result_llama2", "TAM (in million USD)_llama2", "Total Number of Potential Customers (in million)_llama2", "Average Annual Revenue Per Customer (in million USD)_llama2","SAM (in million USD)_llama2",
    "SOM (in million USD)_llama2","Result_llama3", "TAM (in million USD)_llama3", "Total Number of Potential Customers (in million)_llama3", "Average Annual Revenue Per Customer (in million USD)_llama3","SAM (in million USD)_llama3",
    "SOM (in million USD)_llama3","Result_mistral", "TAM (in million USD)_mistral", "Total Number of Potential Customers (in million)_mistral", "Average Annual Revenue Per Customer (in million USD)_mistral","SAM (in million USD)_mistral",
    "SOM (in million USD)_mistral"]


# Write headers to CSV (if the file does not exist or is empty)
csv_file = "4_tam_sam_som_query_result13_6.csv"
try:
    with open(csv_file, 'r', encoding='utf-8') as file:
        pass
except FileNotFoundError:
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def calculate_min_length(text):
    sentences = text.split('.')
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    min_length = int(avg_sentence_length * len(sentences) * 0.5)  # Adjust the factor as needed
    return min_length

def bart_summarize(text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    min_length = calculate_min_length(text)

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=min_length, max_length=500, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

queries = [
 "Calculate the Total Addressable Market (TAM) for the electric vehicle (EV) market in the UAE, Get the total number of potential customers and the average annual revenue per customer. Find TAM from the obtained value of potential customers and annual revenue per customer. Do not give me any calculation. Give me answer in the form of a python dictionary with keywords TC, AR and TAM for Total Number Of Potential Customer, average Annual Revenue Per Customer and TAM respectively and their values in millions.",
#  "Calculate the Total Addressable Market (TAM) for the electric vehicle (EV) market in the KSA, Get the total number of potential customers and the average annual revenue per customer. Find TAM from the obtained value of potential customers and annual revenue per customer. Do not give me any calculation. Give me answer in the form of a python dictionary with keywords TC, AR and TAM for Total Number Of Potential Customer, average Annual Revenue Per Customer and TAM respectively and their values in millions.",
#  "Calculate the Total Addressable Market (TAM) for the electric vehicle (EV) market in the Japan, Get the total number of potential customers and the average annual revenue per customer. Find TAM from the obtained value of potential customers and annual revenue per customer. Do not give me any calculation. Give me answer in the form of a python dictionary with keywords TC, AR and TAM for Total Number Of Potential Customer, average Annual Revenue Per Customer and TAM respectively and their values in millions."
]
# print(len(queries))
for query in queries: 
    result = []
    result.append(query)
    print("JMJ")
    print(query)

    # data = phind_search(query)
    # result.extend([data])
    # tam, potential_customers, revenue_per_customer,sam,som = extract_and_normalize_values(data)
    # result.extend([tam, potential_customers, revenue_per_customer])
    # result.extend([sam, som])

    data = perplexity_search(query)
    print("perplexity running")
    print(data)
    # result.extend([data])
    # tam, potential_customers, revenue_per_customer,sam,som = extract_and_normalize_values(data)
    # result.extend([tam, potential_customers, revenue_per_customer])
    # result.extend([sam, som])


    data = llama2_search(query)
    print("llama2 running")
    print(data)
    # result.extend([data])
    # tam, potential_customers, revenue_per_customer,sam,som = extract_and_normalize_values(data)
    # result.extend([tam, potential_customers, revenue_per_customer])
    # result.extend([sam, som])


    data = llama3_search(query)
    print("llama3 running")
    print(data)
    # result.extend([data])
    # tam, potential_customers, revenue_per_customer,sam,som = extract_and_normalize_values(data)
    # result.extend([tam, potential_customers, revenue_per_customer])
    # result.extend([sam, som])

    
    data = mistral_search(query)
    print("mistral running")
    print(type(data))
    print(data)
    # result.extend([data])
    # tam, potential_customers, revenue_per_customer,sam,som = extract_and_normalize_values(data)
    # result.extend([tam, potential_customers, revenue_per_customer])
    # result.extend([sam, som])
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(result)

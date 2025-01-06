import os
import time
from typing import Literal

import asyncio
import instructor
from openai import AsyncOpenAI
import pandas as pd
from pydantic import BaseModel

# Initialize client
os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_TOKEN_PERSONAL']
client = instructor.from_openai(AsyncOpenAI())

# Define the possible outputs a complaint can have
class ComplaintType(BaseModel):
    complaint: Literal["Bank account or service", "Consumer Loan", "Credit card", "Student loan"]


async def get_prediction(complaint: str) -> str:
    """Query openai to predict the type of transaction that the customer is referring to.

    Parameters
    ----------
    complaint : str
        Customer complaint text

    Returns
    -------
    str
        The complaint classification
    """
    model = "gpt-3.5-turbo"
    prompt = (
        "The following text represents a customer issue or complaint with a financial institution. "
        "The task is to predict the type of transaction that the customer is referring to. "
        "Please classify the following text into one of the following categories: "
        "Bank account or service, Consumer Loan, Credit card, Student loan. \n"
        "-------------------\n"
        "{text}\n" 
    )
    res = await client.chat.completions.create(
        messages = [{
            "role": "user",
            "content": prompt.format(text=complaint)
            }],
        model=model,
        response_model=ComplaintType,
        temperature=0
    )
    return res.complaint

def run_predictions(df: pd.DataFrame, max_wait: int = 60) -> pd.DataFrame:
    """Run predictions on a DataFrame of complaints

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the complaints

    Returns
    -------
    pd.DataFrame
        DataFrame with the predictions
    """
    return asyncio.gather(
            *[get_prediction(complaint) for complaint in df.Consumer_complaint_summary]
    )
    

def main():
    target = 'Transaction_Type'
    predictor = "Consumer_complaint_summary"
    df = pd.read_csv('https://s3.amazonaws.com/datarobot_public/drx/email_training_dataset.csv')[[predictor, target]]
    X_test = df[10000:].reset_index(drop=True)
    preds = []
    for i in range(0, len(X_test.head(300)), 10):
        print(i)
        resp = run_predictions(X_test[i:i+10])
        for i in range(10):
            time.sleep(5)
            print(resp.done())
        time.sleep(10)
        preds.extend(resp.result())
    print(preds)


if __name__ == "__main__":
    main()
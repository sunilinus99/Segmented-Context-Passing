
# Segmented Context Passing

## Why Segmented Context Passing?

### Efficiency:
For the chatbot in our case, not all details are required for every step. When a user uploads a loan request or asks to find a lender, only the relevant fields like the loan amount, credit score, and property location should be passed to the model.

### Token Conservation:
By reducing the amount of data passed with each request, the model focuses on fewer tokens, resulting in more efficient use of token space, lowering overall costs and latency.

## How It Works in Our Use Case:

The chatbot performs three core tasks:
1. Adding the lender to a list.
2. Finding a lender for a housing project.
3. Building a loan request.

At each step, only the relevant data is sent to the model, and past interactions or irrelevant information are not sent unless needed.

I am breaking this down into two essential parts:
- Part 1: Minimal Input from the User
- Part 2: Efficient Data Passing to the Model

## Example Code Implementation

We'll implement a simple system where user data is segmented, and only relevant parts are passed to the model using a FastAPI-based chatbot interaction.

### Part 1: User Interaction and Input

```python
from fastapi import FastAPI

app = FastAPI()

# Dictionary to hold user session data
user_sessions = {}

# Simulate user input for example: Loan Request Details
@app.post("/loan_request/")
def loan_request(user_id: str, loan_amount: float, property_value: float, credit_score: int, location: str):
    # Store user input details in session data (segmented context)
    user_sessions[user_id] = {
        "loan_amount": loan_amount,
        "property_value": property_value,
        "credit_score": credit_score,
        "location": location
    }
    return {"message": "Loan request stored", "session_id": user_id}

# Example function to show segmented data use in the next interaction
@app.get("/find_lender/")
def find_lender(user_id: str):
    if user_id not in user_sessions:
        return {"error": "Session not found"}
    
    # Retrieve only the necessary context to find a lender (segmented)
    loan_details = user_sessions[user_id]
    
    # Pass only necessary fields to model (e.g. loan_amount, credit_score)
    relevant_data = {
        "loan_amount": loan_details["loan_amount"],
        "credit_score": loan_details["credit_score"]
    }
    
    # Placeholder for calling the model or API
    # response = call_openai_model(relevant_data)
    
    return {"message": "Relevant data passed", "data": relevant_data}
```

### Part 2: Efficient Data Passing to the Model

```python
import openai

# Assume we have an OpenAI model
def call_openai_model(data):
    # Construct the prompt with minimal tokens using segmented context
    prompt = f"Find lenders for a loan of ${data['loan_amount']} with a credit score of {data['credit_score']}."
    
    # Example of how to call the OpenAI API (mocking the response)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50  # Minimized token usage
    )
    
    return response.choices[0].text
```

## Explanation of Code

### Storing Context in Sessions:
When the user submits a loan request (`/loan_request/`), the details are saved in a `user_sessions` dictionary.
The user only needs to submit details once, and the chatbot stores this information segmented by categories like `loan_amount`, `credit_score`, `location`, etc.

### Segmenting Data for Lender Match:
When the user asks to find a lender (`/find_lender/`), only the relevant information (loan amount and credit score) is retrieved from the session and passed to the model. This segmented context ensures that only the minimum required information is used, keeping token usage low.

### Optimized Model Prompt:
The prompt passed to the OpenAI model is short and to the point:
"Find lenders for a loan of $400000 with a credit score of 720." This prompt is optimized in token usage, as unnecessary details (like property value, location) are not passed unless needed for a specific use case.

## Benefits of This Approach

1. **Token Efficiency**: By sending only the necessary parts of the conversation or loan details, avoid bloating the input with irrelevant context, drastically reducing the number of tokens.
2. **Dynamic Context Handling**: As the user progresses through different stages (e.g. from finding a lender to creating a loan request), only the most relevant parts are sent to the model at each stage.
3. **Cost and Performance Optimization**: Fewer tokens mean lower costs and faster responses from the model, which is especially important when scaling the chatbot.

## Further Optimizations

- **Stateful Sessions**: We can optimize by using stateful sessions or caching. Once data is stored, we don't need to send it back to the model on every request.
- **Embeddings for Larger Context**: If needed to handle more complex interactions, we can store the overall context in embeddings (vectors), reducing token usage even further when passing back context.

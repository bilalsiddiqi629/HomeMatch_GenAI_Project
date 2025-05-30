import os
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "voc-1766757740126677411696868360f71b23f94.47733535"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"]
)

house_size =input("How big do you want your house to be?\n")
important_things = input("What are 3 most important things for you in choosing this property?\n")
amenities = input("Which amenities would you like?\n")
transportation = input("Which transportation options are important to you?\n")
urban = input("How urban do you want your neighborhood to be?\n")

print("Loading.....\n")
loader = CSVLoader(file_path="./listings.txt")
data = loader.load()

embeddings = OpenAIEmbeddings()



db = Chroma.from_documents(data, embeddings)




def generate_listing(size, amenity, urb, daata, important, transport):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a real estate agent who has been in the industry for many years. You know exactly how much a house should be based on the qualities of said house"},
                
                {"role": "user", "content": 
                 f"""Based on what the user inputted for their preferred size of a house, which is {size}, 
                the amenities included for the house, which is {amenity}, the context of the preferred urbanness of the house 
                given by {urb}, the desired transportation methods readily available given by {transport}, and important factors 
                for a house given by {important}, generate a housing listed based on real world data and the data
                that is provided in {daata}. If there is a mistake in the input, say what it is. Don't return any additional text,
                just the listing. Make sure the listing is in the format:
                
                Price: (estimated price here)
                Bedrooms:  (estimated number of bedrooms)
                Bathrooms:  (estimated number of bathrooms)
                House Size: (estimated house size in square feet)
                Description: (Make a description that encapsulates everything the user inputted)
                Neighborhood Description: (Describe a neighborhood in which this house would exist in using real world context.)
                """}
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"
    
def final_output(most_similar, user):    

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a real estate agent who has been in the industry for many years. You know exactly how much a house should be based on the qualities of said house"},
                
                {"role": "user", "content": 
                 f"""
                 Generate a new listing. Keep the same price, bedroom number, bathroom number, and house size from {most_similar},
                 but tweak the description from {most_similar} so it lines up with the user's desires in {user}. Keep
                 the same neighborhood name of the house from {most_similar}. Make sure it maintains the format below:
                 
                Neighborhood: (insert name here) 
                Price: (estimated price here)
                Bedrooms:  (estimated number of bedrooms)
                Bathrooms:  (estimated number of bathrooms)
                House Size: (estimated house size in square feet)
                Description: (Make a description that encapsulates everything the user inputted)
                Neighborhood Description: (Describe a neighborhood in which this house would exist in using real world context.)
                """}
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


user_generated_listing = generate_listing(house_size, amenities, urban, data, important_things, transportation)
result = db.similarity_search(user_generated_listing, k = 1)
print("Input processed. Generating final output.....\n")

print(final_output(result, user_generated_listing))



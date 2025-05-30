HomeMatch GenAI Project
HomeMatch is a generative AI-powered real estate assistant that recommends the most ideal home listing based on a user's preferences. It leverages OpenAI, LangChain, and ChromaDB to provide intelligent, personalized property matches.

Features
- Converts user preferences into structured search queries
- Uses vector similarity to match homes to user needs
- Enhances listings with OpenAI to better reflect personal taste

Tech Stack
- OpenAI API: Language generation and refinement
- LangChain: Manages prompts and memory
- ChromaDB: The vector database
- Python: Core programming language

How It Works
- Listing Generation
    - OpenAI generates 10 fictional real estate listings with varying features.
- Vector Storage
    - Listings are embedded and stored in ChromaDB via LangChain.
- User Input
    - The user specifies preferences such as house size, desired amenities, neighborhood style, and more.
- Similarity Search
    - The system embeds the user’s input and performs a similarity search in ChromaDB to find the closest match.

Personalized Output
- The best match is refined through OpenAI to tailor the listing description to the user’s specific criteria.

Project Goals
- Apply generative AI to a real-world recommendation task
- Integrate LLMs with vector search using LangChain and ChromaDB
- Build a flexible pipeline for matching user input to structured data

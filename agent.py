import os
from hindsight import HindsightClient
from groq import Groq

# 1. Initialize Clients
hindsight = HindsightClient(api_key=os.environ.get("HINDSIGHT_API_KEY"))
llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def bootstrap_agent_memory():
    """Simulate loading historical client meeting transcripts into long-term memory."""
    past_meetings = [
        "Met with Sarah from Acme Corp on Sept 1. They want to migrate their legacy on-prem SQL database to the cloud. Strict budget cap is $50,000 for phase one.",
        "Follow-up with Sarah on Sept 15. Her engineering team strongly prefers AWS because their lead developer is AWS certified. Deadline is Dec 1 before server lease expires."
    ]
    
    for i, transcript in enumerate(past_meetings):
        hindsight.store_memory(
            content=transcript, 
            metadata={"contact": "Sarah", "company": "Acme Corp", "interaction_id": i}
        )
    print("✓ Historical meeting records embedded into persistent memory.")

def generate_meeting_brief():
    """Query long-term memory to assemble a deeply contextualized briefing."""
    current_prompt = "Prepare an executive briefing for my strategy sync with Sarah from Acme Corp today."
    
    # Retrieve matching context from long-term memory
    relevant_memories = hindsight.query_memory(query="Sarah Acme Corp budget deadline AWS database")
    context_str = "\n".join([mem.content for mem in relevant_memories])

    system_instruction = f"""
    You are an elite enterprise executive assistant. Generate a sharp, highly technical meeting brief.
    Use ONLY the long-term memory context provided below. Do not guess or hallucinate.
    
    CONTEXT FROM PAST INTERACTIONS:
    {context_str}
    
    Structure your response with clear markdown headings:
    # Executive Briefing: Acme Corp Sync
    ## Strategic Pain Points
    ## Hard Technical & Budget Constraints
    ## Recommended Agenda Items
    """

    response = llm.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": current_prompt}
        ]
    )
    
    print("\n=== SYSTEM OUTPUT PREVIEW ===")
    print(response.choices.message.content)

if __name__ == "__main__":
    bootstrap_agent_memory()
    generate_meeting_brief()

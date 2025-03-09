from typing import List, Tuple, Any, Callable

def run_agentic_workflow(
    user_query: str,
    embed_text: Callable[[str], Any],
    retrieve_similar: Callable[[Any, int], Tuple[float, str]],
    response_agent: Any,
    evaluation_agent: Any,
    enrichment_agent: Any,
    decomposition_agent: Any,
    routing_agent: Any,
    aggregation_agent: Any = None,
    similarity_threshold: float = 1.0
) -> str:
    """
    Executes an agentic workflow to process and refine a user query before responding.

    Parameters:
    - user_query (str): The input query from the user.
    - embed_text: Function to convert text into vector representations.
    - retrieve_similar: Function to retrieve the most relevant stored answers.
    - response_agent: Agent responsible for generating responses.
    - evaluation_agent: Agent responsible for assessing the quality of responses.
    - enrichment_agent: Agent that determines if additional information is required.
    - decomposition_agent: Agent that decomposes complex queries into subqueries.
    - routing_agent: Agent that directs subqueries to specialized agents.
    - aggregation_agent (Optional): Agent to consolidate enriched queries into a refined query.
    - similarity_threshold (float, default=1.0): Similarity threshold for returning a stored answer directly.

    Returns:
    - str: The final response from the agentic system.
    """
    
    # Step 1: Retrieve the most relevant stored response
    query_vector = embed_text(user_query)
    similarity_score, best_match_response = retrieve_similar(query_vector, top_k=1)

    # Step 2: If similarity exceeds threshold, return the stored response
    if similarity_score >= similarity_threshold:
        return best_match_response

    # Step 3: Determine if additional enrichment is needed
    refined_query = user_query
    if enrichment_agent.analyze(user_query):  # Assuming this returns a boolean
        
        # Step 4: Refinement process loop
        is_satisfactory = False

        while not is_satisfactory:
            processed_queries = []
            sub_queries = decomposition_agent.decompose(user_query)  # Assuming this returns a list of subqueries

            for sub_query in sub_queries:
                # Step 4.1: Route subquery to the appropriate agent
                specialized_agent = routing_agent.route(sub_query)
                if not specialized_agent:
                    processed_queries.append(sub_query)  # Fallback to original subquery
                    continue  

                preferred_source = specialized_agent.determine_source(sub_query)

                # Step 4.2: Process the subquery through the selected source if available
                try:
                    if preferred_source:
                        tool_inputs = specialized_agent.prepare_request(sub_query, preferred_source)
                        tool_output = preferred_source.execute(tool_inputs)
                        processed_query = specialized_agent.process_response(sub_query, tool_output, preferred_source)
                    else:
                        processed_query = sub_query
                except Exception as e:
                    processed_query = sub_query  # Fallback in case of failure
                    print(f"Error processing sub_query '{sub_query}': {e}")

                processed_queries.append(processed_query)

            # Step 5: Aggregate processed queries if an aggregation agent is provided
            if aggregation_agent:
                refined_query = aggregation_agent.aggregate(processed_queries)
            else:
                refined_query = " ".join(processed_queries)  # Default behavior: merge queries

            # Step 6: Evaluate the refined query
            is_satisfactory, feedback = evaluation_agent.evaluate(refined_query)  # Expecting (bool, List[Tuple[str, Any]])

            # Step 7: If unsatisfactory, refine strategies based on feedback
            if not is_satisfactory:
                for critique, agent in feedback:
                    agent.improve_strategy(critique)

    # Step 8: Generate final response
    return response_agent.generate_response(refined_query)

### LLM.py

These are my notes from the Deep Learning AI course by Andrew Ng on Coursera.

Here's what I've learned so far:

### Text Completion
- **Simple Text Completion**: Use `ChatOpenAI`.

### Prompt Management
- **Reuse Prompts**: Modify and parameterize the prompt yourself.
- **Long and Detailed Prompts**: Use `Prompt Templates` to auto-concatenate arguments.

### JSON Handling
- **Return as JSON Dict**: Instruct the prompt to return a JSON dict and (try to) parse it.
- **Ensure JSON Format**: Use `Response Schema` to auto-generate format instructions.

### Memory Management
- **Stateless LLM**: Use `Conversation Buffer Memory` to maintain context.
- **Modifiable Buffer**: Load memory variables and manually save context.
- **Growing Buffer**: Use `Buffer Window Memory` to store a fixed number of answers.
- **Large Answers**: Use `Token Window Memory` to limit to a specific number of words.
- **Token Optimization**: Use `Summary Buffer Memory` for better efficiency and higher performance.
- **Relevant Texts**: Use `Vector Data Memory` to store only relevant texts.
- **Specific Facts/Entities**: Use `Entity Memories` for specific information.
- **Combining Approaches**: Use multiple memory types simultaneously.
- **Caching**: Store conversations in a conventional database like a key-value store or SQL.

### Chain Management
- **Sequential I/O Processes**: Save chains in a `Sequential Chain`.
- **Conditional Logic**: Save chains in a `Router Chain`.

### Database Querying
- **Querying**: Run queries on an `Index Vector Store`.


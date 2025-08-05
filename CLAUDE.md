# Claude Instructions and Guidelines

This file contains important instructions and guidelines for working on this project.

## Testing Guidelines

### Test Complexity and Scope
- **NEVER make tests simpler unless explicitly asked**
- Keep comprehensive tests even if they require additional setup
- Don't remove test functionality without explicit user approval
- Wait for explicit user response before simplifying tests

### Handling Test Dependencies and Issues
- When tests fail due to missing libraries or dependencies, use standard software engineering approaches to fix them
- If the approach is standard and doesn't create technical debt (e.g., creating requirements.txt, setting up virtual environments), proceed with the fix
- If the best choice is ambiguous, ask for guidance and provide multiple options
- Examples of standard approaches:
  - Creating comprehensive requirements.txt files
  - Setting up virtual environments
  - Installing missing dependencies
  - Configuring proper Python paths

## Code Quality and Architecture

### Dependencies and Environment Management
- Always maintain proper dependency management
- Use virtual environments for isolated development
- Document all required dependencies clearly
- Include both main dependencies and sub-project dependencies

### Error Handling and Debugging
- Don't hide or work around import errors by simplifying code
- Fix the root cause of dependency issues
- Maintain full functionality even when it requires additional setup

## General Development Principles

### When to Ask vs. Proceed
- **Proceed without asking when:**
  - Using standard software engineering practices
  - Fixing clear technical issues with established solutions
  - Following well-known conventions
  
- **Ask before proceeding when:**
  - Multiple approaches are equally valid
  - The solution might impact architecture significantly
  - Unsure about user preferences for specific implementations
  - Changes might affect existing functionality in unclear ways

### Documentation and Communication
- Always explain what standard approaches are being used and why
- Document decisions and rationale
- Keep this CLAUDE.md file updated with new instructions
- Reference this file when following established guidelines

## Project-Specific Guidelines

### Pipeline Integration
- Maintain compatibility between yochlol and tabular_modeling components
- Preserve original functionality while adding orchestration
- Use cohort configuration system for all environment-specific settings

### Testing Strategy
- Test integration points between components
- Validate data format compatibility
- Test cohort configuration system thoroughly
- Include end-to-end integration tests

## Session History and Progress Tracking

### History File Management
- **ALWAYS update the CODE_HISTORY.md file** after any significant work or when completing tasks
- Include what was accomplished, what works robustly, what still needs testing
- Document user intentions and context for future sessions
- Note any unresolved issues or areas needing attention
- This helps maintain continuity across disconnected sessions

### When to Update History
- After implementing major features or fixes
- When completing a significant milestone  
- Before ending a session or when conversation context is getting full
- When user provides important context or feedback
- After resolving complex technical issues

---

**Note**: This file should be referenced for guidance on project decisions and updated when new instructions are provided.
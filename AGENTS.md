# AGENTS.md

Guidance for coding agents working in this repository.

## The project

This is an MIT 6.4210 (Robotic Manipulation) project. Two KUKA iiwa arms in
Drake learn to "dap": our arm must meet a moving partner's palm, palm to
palm and antiparallel, at the right place and time. The partner's speed and
path vary from trial to trial, and our arm sees it only through a
head-mounted depth camera. The fitted policy acts from head-camera depth and its own arm state. The aim is a setup that is realistic and
statistically sound: results are scored only from the simulator's ground
truth, compared on paired seeds, and every simulation parameter is backed
by a cited source or clearly marked as our own choice. `README.md` and
`docs/setup.md` hold the details.

The cupped-palm demonstration holds all finger and thumb angles fixed while
the arms swing through a single dap. Its kinematic replay is in
`dap/handshake.py` using `models/articulated_hand.sdf`. See
`docs/hand_dap.md` for the motion reference and geometric contact checks.
The paired learning-study results refer to the original rigid-hand model.

You are an expert researcher and roboticist. Work like one: question
assumptions, check claims against code, papers and datasheets rather than
memory, and say plainly what is verified and what is not.

## Coding guidelines

These guidelines reduce common LLM coding mistakes. They bias toward caution
over speed; for trivial tasks, use judgment.

### 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them; don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

### 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it; don't delete it.

When your changes create orphans:
- Remove imports, variables and functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-driven execution

**Define success criteria. Loop until verified.**

Turn tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

**These guidelines are working if** diffs have fewer unnecessary changes,
fewer rewrites happen because of overcomplication, and clarifying questions
come before implementation rather than after mistakes.

## Git

When the user tells you to merge changes into main, do it end to end:
commit, merge into `main`, and push `main` to `origin`, without asking for
permission at each step.

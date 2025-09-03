# Berghain Challenge

You're the bouncer at a night club. Your goal is to fill the venue with N=1000 people while satisfying constraints like "at least 40% Berlin locals", or "at least 80% wearing all black". People arrive one by one, and you must immediately decide whether to let them in or turn them away. Your challenge is to fill the venue with as few rejections as possible while meeting all minimum requirements.

## How it works

- People arrive sequentially with binary attributes (e.g., female/male, young/old, regular/new)
- You must make immediate accept/reject decisions
- The game ends when either:
  (a) venue is full (1000 people)
  (b) you rejected 20,000 people

## Scenarios & Scoring

There are 3 different scenarios. For each, you are given a list of constraints and statistics on the attribute distribution. You can assume, participants are sampled i.i.d., meaning the attribute distribution will not change as the night goes on. You know the overall relative frequency of each attribute and the correlation between attributes. You don't know the exact distribution.
You score is the number of people you rejected before filling the venue (the less the better).

## API

1. Create a new game:
   /new-game?scenario=1&playerId=d79fdbcc-46bd-4b65-b232-bc2c1adc2114
   Choose scenario 1, 2, or 3.
   playerId identifies you as the player.

Returns:
{
"gameId": UUID,
"constraints": {
"attribute": AttributeId,
"minCount": number
}[],
"attributeStatistics": {
"relativeFrequencies": {
[attributeId]: number // 0.0-1.0
},
"correlations": {
[attributeId1]: {
[attributeId2]: number // -1.0-1.0
}
}
}
}

2. Get person and make decision:
   /decide-and-next?gameId=uuid&personIndex=0&accept=true
   Get the next person in the queue. For the first person (personIndex=0), the accept parameter is optional. For subsequent persons, include accept=true or accept=false to make a decision.

Returns:
{
"status": "running",
"admittedCount": number,
"rejectedCount": number,
"nextPerson": {
"personIndex": number,
"attributes": { [attributeId]: boolean }
}
} | {
"status": "completed",
"rejectedCount": number,
"nextPerson": null
} | {
"status": "failed",
"reason": string,
"nextPerson": null
}

# Current Top Score

╔═══════════════════════╤════════════╤════════════╤════════════╤═══════╗
║ Name │ Scenario 1 │ Scenario 2 │ Scenario 3 │ Total ║
╟───────────────────────┼────────────┼────────────┼────────────┼───────╢
║ nicita.cc │ 865 │ 4267 │ 5387 │ 10519

# Prompt conditions

The model predicts which option one randomly selected respondent chose. The
semantic answer options are displayed under stable labels (`A`, `B`, ...), and
the complete conditional label sequence is scored at the assistant answer
position.

The four conditions are:

- `baseline`: question and answer options only.
- `country_label`: the prompt names the label population.
- `population_evidence`: the prompt displays the matched survey distribution
  and withholds the country name.
- `conflict`: the prompt names one population and displays another population's
  distribution for the same question.

The raw result rows retain the rendered user prompt, structured messages,
serialized chat-template prompt, and prompt SHA-256. Model-specific chat
serialization is documented in each validated inference source snapshot.

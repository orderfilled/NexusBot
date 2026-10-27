# Research candidate used by the weekly forward test

`model.pt` is `local_20261027_001810`: trained on Jan 27 to Jun 30 2026 XAUUSD M1
bars. It FAILED the bot's behavioural integrity gate (DEGENERATE: logit_std and
sensitivity too low), so the bot will never serve it. It is kept only as the
fixed model for measuring the rules + probability gate. Do not deploy it.

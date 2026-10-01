# IonQ fixture

`responses.json` maps IonQ API v0.4 URLs to response bodies shaped like the real ones, with
made-up numbers and a 4-qubit `qpu.forte-1`. It reproduces what the live endpoint does: null
and placeholder fields (SPAM fidelity 0.501, stderr 0), and pages of several records in which a
record's null field is filled from the next newer record on that page, while a one-record page
shows the record's own values.

It also holds the kinds of records IonQ's history contains that must not become profiles: a
`qpu.aria-1` record with an implausible 1Q median (0.79) next to a normal 2Q median, and a retired
`qpu.harmony` whose records never carry a 1Q fidelity.

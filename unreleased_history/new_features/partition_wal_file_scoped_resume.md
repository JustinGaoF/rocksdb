# Live resume after a file-scoped WAL write error with partitioned WALs

With `wal_partitions > 1`, a file-scoped WAL write error (`kIOErrorScopeFile`)
no longer ends the DB. `Resume()`, and auto-resume, now bring the DB back to
accepting writes on every partition without a reopen: every partition rotates
to a replacement WAL, every live column family is atomically flushed past the
failed WAL, every WAL at or below the failed one is retired in memory and in
the MANIFEST WAL set, and the failed ranges are released.

The failed write, and every write fenced above it, stay invisible before and
after the resume and after any reopen that follows it, in every recovery
mode. No sequence or WAL index they allocated is ever reused, and a failed
sync-led group's ranges release exactly as they were registered.

Configurations the resume still refuses (`manual_wal_flush`, 2PC, two write
queues, pipelined or unordered writes, WAL recycling, retention or tracking
(`track_and_verify_wals`), and blob direct write), and single-partition DBs
with MANIFEST WAL tracking, behave exactly as before.

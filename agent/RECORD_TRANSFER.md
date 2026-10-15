# Stored record transfer

`RexClient` transfers selected RCDB record versions through `/rex/v1`. Record
packets support graphs, temporal objects, exact values, declarations, provenance
and RCQL results when their codec is installed. The application supplies a peer
token, workspace and local store:

```python
from agent.client import RexClient
from rcdb import record_packet

peer = RexClient("https://team-server:8000", api_key=token, workspace="research")
receipt = peer.rex_store_record(record_packet(local_store, "reading", version=1))
packet = peer.rex_fetch_record(receipt.destination_record_id,
                               version=receipt.destination_version)
snapshot = packet.snapshot()
```

The receiver publishes through `rcdb.copy_record`, applies workspace ownership
and destination policy, and returns a checked binary receipt. Payload values,
metadata and validity retain their declared values; the destination allocates
transaction clocks and applies its own encryption and mutation policy. The
transport supports optional frame HMAC authentication and bounded body streams.

Remote couriers use record packets when `/rex/v1/hello` advertises
`record_transfer_version=1`. Older peers support graph transfers only. Couriers
detect metadata changes as well as payload changes. See the
[RCDB packet contract](../rcdb/README.md#portable-record-versions).

## Sender acknowledgements

`Ledger(path)` stores successful transfer receipts. A shared file lock serializes
the shipment decision, acknowledgement and ledger publication across independent
handles and POSIX processes. Separate ledgers proceed independently. Open fresh
courier, ledger and store handles after a fork.

Malformed ledgers reject loading. An uncertain ledger write blocks reuse until
`ledger.load()` verifies the published ledger. That check validates local
receipts; reconciliation checks a selected destination version.

If remote publication succeeds but acknowledgement fails, the shipment reports
`refused` and retains the known destination address and checked receipt. Use
explicit reconciliation to verify the destination and record the acknowledgement:

```python
from rcdb import CopyReceipt, record_packet

# remote_peer is a registered agent.courier_remote.Peer; failed is its Shipment.
receipt = CopyReceipt.from_record(failed.receipt)
selected = record_packet(local_store, failed.record_id, version=receipt.source_version)
reconciled = remote_peer.reconcile(selected, receipt)
```

Reconciliation fetches the destination version through `RexClient`, checks store
identities, record addresses, versions and state digests, then records the receipt.
It returns `held` without publishing another record. The admin route
`POST /api/v1/courier/reconcile` accepts `{source, dest, record_id, receipt}` for
registered stores and peers under the workspace and audit policy. Delivery,
broadcast and reconciliation run in FastAPI worker threads, including calls to
peers on the same server.

A lost response without a receipt or destination address requires operator
recovery. Transfers do not provide exactly once delivery. Reconciling an older
selected version can replace a newer acknowledgement for that source record.

## Concurrent publication and outcomes

Local couriers use `rcdb.copy_record` with the destination publication cursor and
policy digest. Concurrent copies of the same selected version publish one
destination version; subsequent trips verify it and report `held`. A change to
that destination record, including deletion and recreation, reports `conflict`
and leaves it intact.

When publication reports a failed precondition and only unrelated records have
changed, the courier can refresh the cursor and retry. `Courier(copy_attempts=4)`
defaults to four attempts and accepts values from 1 through 32. A changed policy
or a destination that cannot retain selection identity rejects publication.
Legacy writers retain their own consistency behavior.

A failed write reports `refused`; an unknown publication outcome reports
`uncertain` and requires reopening and verification. Source records that cannot
be read are counted as `unreadable`. The receipt's `parent_version` identifies
the live predecessor, or None when recreating a deleted record.

Each trip selects routes and versions before I/O, so route changes apply on the
next trip. Selection accepts literal record ID lists and a nonnegative limit.
`limit=0` selects nothing, and explicit empty filters override configured filters.
The caller manages the registered store handles.

## Sender ledger retention

`Ledger` keeps the latest acknowledgement for each peer and record address.
To retire inactive entries, create a plan with
`plan_retention(before=..., peer=..., max_entries=...)`, then call
`apply_retention(plan, archive_path=...)`. The archive contains complete receipts
and is published before removal. A changed ledger rejects a stale plan.

Removing an acknowledgement can permit a later resend. Archived receipts remain
available for reconciliation; an uncertain acknowledgement requires verified
reload before the ledger is reused.

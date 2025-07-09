from typing import Any
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import (
    DatabaseInstance,
    SyncedTableSpec,
    SyncedDatabaseTable,
    NewPipelineSpec,
    SyncedTableSchedulingPolicy,
)
from databricks.sdk.errors import NotFound
import pyperclip


@dataclass
class TableMapping:
    """
    Represents a mapping between a source table and a destination table.
    """
    source_table: str
    destination_table: str
    primary_key: str
    time_key: str | None = None


# Set the names and catalog/schema for the lakebase instance and tables.
LAKEBASE_INSTANCE_NAME = "youtube-lakebase-instance"
CATALOG = "mk_fiddles"
SCHEMA = "youtube"

# Define the mappings for the tables to be synced
statistics = TableMapping(
    source_table="statistics_gold",
    destination_table="statistics",
    primary_key="video_id",
    time_key="timestamp",
)
metadata = TableMapping(
    source_table="metadata_silver",
    destination_table="metadata",
    primary_key="video_id",
)
# Initialize the WorkspaceClient
# Set your configuration if you're not already logged in
w = WorkspaceClient()

def get_or_create_lakebase_instance(w: WorkspaceClient) -> dict[str, Any]:
    """
    Return lakebase instance if name matches or create a new one.
    """

    try:
        return w.database.get_database_instance(LAKEBASE_INSTANCE_NAME)
    except NotFound:
        database_instance = DatabaseInstance(
            name=LAKEBASE_INSTANCE_NAME, capacity="CU_1", stopped=False
        )
        db = w.database.create_database_instance(database_instance=database_instance)

        return db


def create_synced_table(
    w: WorkspaceClient, table_mapping: TableMapping
) -> SyncedDatabaseTable:
    """
    Create a synced table in the lakebase instance based on the provided table mapping.

    The create call is idempotent, meaning if the table already exists,
    it will not create a new one but return the existing synced table.
    """
    full_source_table_name = f"{CATALOG}.{SCHEMA}.{table_mapping.source_table}"
    full_destination_table_name = (
        f"{CATALOG}.{SCHEMA}.{table_mapping.destination_table}"
    )

    sync_db_table = SyncedDatabaseTable(
        name=full_destination_table_name,
        database_instance_name=LAKEBASE_INSTANCE_NAME,
        logical_database_name=CATALOG,
        spec=SyncedTableSpec(
            new_pipeline_spec=NewPipelineSpec(
                storage_catalog=CATALOG,
                storage_schema=SCHEMA,
            ),
            scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
            source_table_full_name=full_source_table_name,
            primary_key_columns=[table_mapping.primary_key],
            timeseries_key=table_mapping.time_key,
        ),
    )
    print("Creating synced table:", full_destination_table_name)
    synced_table = w.database.create_synced_database_table(sync_db_table)

    return synced_table

def run_sync_pipelines(
    w: WorkspaceClient,
    pipeline_ids: list[str],
) -> None:
    """
    Sync multiple tables based on the provided table mappings.
    """

    print("Updating pipelines for statistics and metadata tables...")

    for pipeline_id in pipeline_ids:
        w.pipelines.start_update(pipeline_id)
    for pipeline_id in pipeline_ids:
        w.pipelines.wait_get_pipeline_idle(pipeline_id)


def copy_lakebase_token_to_clipboard(w: WorkspaceClient, request_id: str) -> None:
    """
    Generate a database credential for the lakebase instance and copy it to the clipboard.
    """
    credential = w.database.generate_database_credential(
        instance_names=[LAKEBASE_INSTANCE_NAME],
        request_id=request_id,
    )
    pyperclip.copy(credential.token)
    print("Lakebase instance created with credentials copied to clipboard.")
    print(
        "Paste the token in postgres connection string to connect to the lakebase instance."
    )


if __name__ == "__main__":
    
    database = get_or_create_lakebase_instance(w)

    statistics_response = create_synced_table(w, statistics)
    metadata_response = create_synced_table(w, metadata)

    run_sync_pipelines(
        w,
        pipeline_ids=[
            statistics_response.data_synchronization_status.pipeline_id,
            metadata_response.data_synchronization_status.pipeline_id,
        ],
    )

    copy_lakebase_token_to_clipboard(w, request_id="lakebase-from-uc-clip-token")
    

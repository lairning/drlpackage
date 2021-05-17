import json

from lairningdecisions.utils.db import db_connect, BACKOFFICE_DB_NAME, SQLParamList


def recreate_db():
    db = db_connect(BACKOFFICE_DB_NAME)

    try:
        db.execute("drop table if exists trainer_cluster")
        db.execute("drop table if exists policy")
        db.execute("drop table if exists listofvalues")

    except Exception as e:
        raise e

    # An entry for each trainer cluster. Stop will be filled with the remove operation.
    db.execute('''create table trainer_cluster
                  (id INTEGER PRIMARY KEY,
                   name unicode,
                   cloud_provider unicode,
                   last_start TIMESTAMP,
                   status unicode,
                   config json
                   )''')

    # An entry for each policy deployed. It may associated (or not) with an endpoint.
    db.execute('''create table policy
                   (trainer_id INTEGER,
                    policy_id INTEGER,
                    backend_name unicode,                 
                    data json,
                    PRIMARY KEY(trainer_id, policy_id),
                    FOREIGN KEY(trainer_id) REFERENCES trainer_cluster(id) ON DELETE CASCADE
                    )''')

    # An entry for each policy deployed. It may associated (or not) with an endpoint.
    db.execute('''create table listofvalues
                   (lov_id INTEGER PRIMARY KEY,
                    lov_name unicode,
                    lov_value_id INTEGER,
                    lov_value_name unicode,
                    data json
                    )''')

    db.commit()


def load_data():
    db = db_connect(BACKOFFICE_DB_NAME)

    # Insert Cloud Cluster Configurations
    data = [
        ('azure_config', 1, 'config_smal',
         json.dumps({'worker_nodes': 2, 'header_type': 'Standard_D4s_v3', 'worker_type': 'Standard_D2s_v3'})),
        ('azure_config', 2, 'config_large',
         json.dumps({'worker_nodes': 1, 'header_type': 'Standard_D8s_v3', 'worker_type': 'Standard_D2s_v3'})),
        ('azure_config', 3, 'config_gpu',
         json.dumps({'worker_nodes': 16, 'header_type': 'Standard_NC6', 'worker_type': 'Standard_D4s_v3'})),
        ('aws_config', 1, 'config_smal',
         json.dumps({'worker_nodes': 2, 'header_type': 'm5.2xlarge', 'worker_type': 'm5.large'})),
        ('aws_config', 2, 'config_large',
         json.dumps({'worker_nodes': 16, 'header_type': 'm5.2xlarge', 'worker_type': 'm5.xlarge'})),
        ('aws_config', 3, 'config_gpu',
         json.dumps({'worker_nodes': 16, 'header_type': 'p2.xlarge', 'worker_type': 'm5.xlarge'})),
    ]
    sql = "INSERT INTO listofvalues (lov_name, lov_value_id, lov_value_name, data) VALUES ({})".format(SQLParamList(4))
    cursor = db.cursor()
    cursor.executemany(sql, data)

    db.commit()


if __name__ == "__main__":
    recreate_db()
    load_data()

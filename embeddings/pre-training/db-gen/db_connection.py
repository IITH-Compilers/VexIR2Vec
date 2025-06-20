# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Used for connection of the database"""

import sqlalchemy as db
from sqlalchemy.orm import scoped_session, sessionmaker
from binary_db_model import BinaryMetaData
import pickle

engine = db.create_engine("sqlite:////Pramana/VexIR2Vec/VexirDB/coreutils.db")
session = sessionmaker(bind=engine)
connection = session()
result = (
    connection.query(BinaryMetaData)
    .filter(BinaryMetaData.id == "coreutils_x86-gcc-6-O2_stripped_sync.out")
    .all()
)

for row in result:
    print(row.id)
    print(pickle.loads(row.ext_lib_functions))
    print(pickle.loads(row.string_func_embedding))

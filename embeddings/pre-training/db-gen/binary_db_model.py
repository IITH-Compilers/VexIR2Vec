# Part of the VexIR2Vec Project, under the AGPL V3.0 License. See the
# LICENSE file or <https://www.gnu.org/licenses/> for license information.

"""Required to initialise the database"""

from sqlalchemy import Column, Integer, String, BLOB, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import OperationalError, NoResultFound, MultipleResultsFound


Model = declarative_base(name="Model")


class binaryMetaData(Model):
    __tablename__ = "binary_metadata"
    id = Column("binary_id", String(50), primary_key=True)
    ext_lib_functions = Column(BLOB)
    string_func_embedding = Column(BLOB)
    isCalled = Column(BLOB)
    func_callee_edges = Column(BLOB)
    cgv_nodes = Column(BLOB)

    def __init__(
        self,
        id,
        ext_lib_functions,
        string_func_embedding,
        isCalled,
        func_callee_edges,
        cgv_nodes,
    ):
        self.id = id
        self.ext_lib_functions = ext_lib_functions
        self.string_func_embedding = string_func_embedding
        self.isCalled = isCalled
        self.func_callee_edges = func_callee_edges
        self.cgv_nodes = cgv_nodes


class binaryMetaDataUnstripped(Model):
    __tablename__ = "binary_metadata_unstripped"
    id = Column("binary_id", String(50), primary_key=True)
    addr_to_line = Column(BLOB)
    cgv_nodes = Column(BLOB)

    def __init__(self, id, addr_to_line, cgv_nodes):
        self.id = id
        self.addr_to_line = addr_to_line
        self.cgv_nodes = cgv_nodes

        # RUN ONCE FOR TABLE CREATION


def startDb(db_path):
    def initDb():
        Model.metadata.create_all(bind=engine)

    engine = create_engine(
        f"sqlite:///{db_path }",
        pool_size=10,
        max_overflow=20,
        connect_args={"timeout": 20},
    )
    session = sessionmaker(bind=engine)
    connection = session()
    try:
        res = connection.query(binaryMetaData).one()
        del res
    except OperationalError:
        initDb()
    except NoResultFound:
        pass
    except MultipleResultsFound:
        pass
    except Exception:
        pass
    engine.dispose()
    connection.close()
    return


def startDbUnstripped(db_path):
    def initDb():
        Model.metadata.create_all(bind=engine)

    engine = create_engine(
        f"sqlite:///{db_path }",
        pool_size=10,
        max_overflow=20,
        connect_args={"timeout": 20},
    )
    session = sessionmaker(bind=engine)
    connection = session()
    try:
        res = connection.query(binaryMetaDataUnstripped).one()
        del res
    except OperationalError:
        initDb()
    except NoResultFound:
        pass
    except MultipleResultsFound:
        pass
    engine.dispose()
    connection.close()
    return

"""
数据库配置文件

支持多种数据库：MySQL、PostgreSQL、SQLite
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# =============================================================================
# 数据库类型选择
# =============================================================================

DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # 可选: sqlite, mysql, postgresql

# =============================================================================
# 数据库连接配置
# =============================================================================

# SQLite 配置（默认，用于开发和测试）
SQLITE_CONFIG = {
    "url": "sqlite:///./app.db",
    "connect_args": {"check_same_thread": False},
    "echo": False,
}

# MySQL 配置
MYSQL_CONFIG = {
    "url": (
        f"mysql+pymysql://"
        f"{os.getenv('MYSQL_USER', 'root')}:"
        f"{os.getenv('MYSQL_PASSWORD', 'password')}@"
        f"{os.getenv('MYSQL_HOST', 'localhost')}:"
        f"{os.getenv('MYSQL_PORT', '3306')}/"
        f"{os.getenv('MYSQL_DATABASE', 'test_db')}"
    ),
    "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "echo": os.getenv("DB_ECHO", "false").lower() == "true",
}

# PostgreSQL 配置
POSTGRESQL_CONFIG = {
    "url": (
        f"postgresql://"
        f"{os.getenv('POSTGRES_USER', 'postgres')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'password')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DATABASE', 'test_db')}"
    ),
    "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
    "pool_pre_ping": True,
    "echo": os.getenv("DB_ECHO", "false").lower() == "true",
}

# =============================================================================
# 根据环境变量选择数据库配置
# =============================================================================

def get_database_config():
    """根据 DB_TYPE 环境变量返回对应的数据库配置"""
    if DB_TYPE == "mysql":
        return MYSQL_CONFIG
    elif DB_TYPE == "postgresql":
        return POSTGRESQL_CONFIG
    else:  # 默认使用 SQLite
        return SQLITE_CONFIG


# =============================================================================
# 创建数据库引擎
# =============================================================================

def create_database_engine():
    """创建数据库引擎"""
    config = get_database_config()
    
    if DB_TYPE == "sqlite":
        engine = create_engine(
            config["url"],
            connect_args=config.get("connect_args", {}),
            echo=config.get("echo", False)
        )
    else:
        engine = create_engine(
            config["url"],
            pool_size=config.get("pool_size", 10),
            max_overflow=config.get("max_overflow", 20),
            pool_pre_ping=config.get("pool_pre_ping", True),
            pool_recycle=config.get("pool_recycle", 3600),
            echo=config.get("echo", False)
        )
    
    return engine


# =============================================================================
# 创建会话工厂
# =============================================================================

engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# =============================================================================
# 依赖项：获取数据库会话
# =============================================================================

def get_db():
    """
    FastAPI 依赖项：获取数据库会话
    
    用法：
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# 工具函数
# =============================================================================

def init_database():
    """初始化数据库（创建所有表）"""
    Base.metadata.create_all(bind=engine)
    print(f"✓ 数据库已初始化（{DB_TYPE}）")


def drop_database():
    """删除所有表（谨慎使用！）"""
    Base.metadata.drop_all(bind=engine)
    print(f"✓ 所有表已删除（{DB_TYPE}）")


def get_database_info():
    """获取数据库连接信息"""
    config = get_database_config()
    
    return {
        "type": DB_TYPE,
        "url": config["url"].split("@")[-1] if "@" in config["url"] else config["url"],
        "pool_size": config.get("pool_size"),
        "max_overflow": config.get("max_overflow"),
    }


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("数据库配置信息：")
    print("-" * 50)
    
    info = get_database_info()
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    
    print("-" * 50)
    
    # 测试连接
    try:
        with engine.connect() as conn:
            print("✓ 数据库连接成功！")
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")

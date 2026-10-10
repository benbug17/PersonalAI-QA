import sqlite3
import pytest
import db

pytestmark = pytest.mark.unit


def test_password_is_hashed_not_stored_plain():
    h = db.hash_password("secret123")
    assert h != "secret123" and h.startswith("$2")
    assert db.verify_password("secret123", h)
    assert not db.verify_password("wrong", h)


def test_hash_is_salted():
    assert db.hash_password("same") != db.hash_password("same")


def test_create_and_authenticate_user():
    uid = db.create_user("bob", "secret123")
    assert isinstance(uid, int)
    assert db.authenticate_user("bob", "secret123") == uid
    assert db.get_username(uid) == "bob"


def test_duplicate_username_returns_none():
    assert db.create_user("bob", "secret123")
    assert db.create_user("bob", "other-pass") is None


def test_authenticate_wrong_password_and_unknown_user():
    db.create_user("bob", "secret123")
    assert db.authenticate_user("bob", "nope") is None
    assert db.authenticate_user("ghost", "secret123") is None


def test_history_roundtrip_newest_first_and_limit():
    uid = db.create_user("bob", "secret123")
    for i in range(5):
        db.save_conversation(uid, f"q{i}", f"a{i}")
    items = db.get_user_history(uid, limit=3)
    assert [i["query"] for i in items] == ["q4", "q3", "q2"]
    assert set(items[0]) == {"id", "query", "response", "timestamp"}


def test_history_is_per_user():
    a, b = db.create_user("a_user", "secret123"), db.create_user("b_user", "secret123")
    db.save_conversation(a, "only-a", "x")
    assert db.get_user_history(b) == []


def test_get_username_unknown():
    assert db.get_username(9999) is None


def test_passwords_never_stored_in_plaintext():
    db.create_user("bob", "super-secret-pw")
    raw = sqlite3.connect(db.DB_PATH).execute("SELECT password_hash FROM users").fetchone()[0]
    assert "super-secret-pw" not in raw and raw.startswith("$2")


def test_init_db_is_idempotent_and_keeps_data():
    uid = db.create_user("bob", "secret123")
    db.init_db(); db.init_db()
    assert db.get_username(uid) == "bob"


def test_sql_metacharacters_are_stored_literally():
    evil = "x'); DROP TABLE users;--"
    uid = db.create_user(evil, "secret123")
    assert db.get_username(uid) == evil and db.authenticate_user(evil, "secret123") == uid


def test_unicode_roundtrip():
    uid = db.create_user("josé", "secret123")
    db.save_conversation(uid, "¿Qué es la fotosíntesis? 🌱", "Es el proceso de las plantas.")
    assert db.get_user_history(uid)[0]["query"] == "¿Qué es la fotosíntesis? 🌱"


def test_concurrent_registration_of_same_username_creates_exactly_one_user():
    import threading
    results = []
    def worker():
        results.append(db.create_user("racer", "secret123"))
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(results) == 8
    assert sum(r is not None for r in results) == 1

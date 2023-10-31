"""性能計測を行うためのモジュール"""
from contextlib import contextmanager, nullcontext
import time


global _with_nest
_with_nest = 0


@contextmanager
def count_and_print_context(name):
    """withブロックで囲まれた範囲の実行時間を終了時に表示するコンテキスト"""
    global _with_nest
    start = time.time()
    _with_nest += 1
    yield
    _with_nest -= 1
    nesting_str = '  ' * _with_nest
    print(f"[perf] {time.time() - start:.08f} [{nesting_str}{name}]")


def get_timer(name, is_count):
    """is_countがTrueの場合、実行時間計測用のコンテキストを取得する"""
    if is_count:
        return count_and_print_context(name)
    else:
        return nullcontext()

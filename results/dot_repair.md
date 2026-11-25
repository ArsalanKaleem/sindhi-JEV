# Dot repair (synthetic)

Margin 0.6 chosen on val_hard (max fixed - broken); test sets use it unchanged.

| set | n | wrong before | exact before | exact after | fixed | broken |
|---|---|---|---|---|---|---|
| test_hard | 2000 | 1000 | 0.500 | 0.990 | 980 | 1 |
| test_font_hard | 2000 | 1000 | 0.500 | 0.933 | 869 | 2 |
| test_iid | 4000 | 1918 | 0.520 | 0.723 | 809 | 1 |
| test_font | 4000 | 1976 | 0.506 | 0.693 | 749 | 1 |

val_hard sweep: m=0.0: +980/-3, m=0.05: +980/-3, m=0.1: +979/-3, m=0.2: +978/-3, m=0.3: +978/-3, m=0.4: +978/-2, m=0.5: +978/-2, m=0.6: +978/-1, m=0.7: +977/-1

import csv, sys
from collections import defaultdict
def report(path):
    rows=list(csv.DictReader(open(path, newline="")))
    reg=defaultdict(float); regu=defaultdict(int); mon=defaultdict(float); prod=defaultdict(float); total=0.0
    for r in rows:
        rev=int(r["units"])*float(r["unit_price"]); total+=rev
        reg[r["region"]]+=rev; regu[r["region"]]+=int(r["units"]); mon[r["date"][:7]]+=rev; prod[r["product"]]+=rev
    out=[f"Sales report: {len(rows)} rows, {len(reg)} regions, {len(prod)} products", f"Total revenue: {total:.2f}", "", "By region:"]
    for k in sorted(reg, key=lambda k:(-reg[k],k)): out.append(f"  {k} {reg[k]:.2f} ({regu[k]} units)")
    out+=["", "By month:"]
    for k in sorted(mon): out.append(f"  {k} {mon[k]:.2f}")
    top=sorted(prod, key=lambda k:(-prod[k],k))[0]
    out+=["", f"Top product: {top} ({prod[top]:.2f})"]
    return "\n".join(out)+"\n"
if __name__=="__main__": sys.stdout.write(report(sys.argv[1]))

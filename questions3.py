import pandas as pd

decalage=3

df = pd.DataFrame({'value(t)':range(100)})

value = df['value(t)']
value = value.shift(-decalage)

df['value(t+{0})'.format(decalage)] = value
df=df[:-decalage]
df = df.round(0).astype(int)#why the type has been changed to float by the method "shift"?

print(df)


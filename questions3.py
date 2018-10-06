import pandas as pd

decalage=3

df = pd.DataFrame({'value(t)':range(100)})

#df['value(t+2)'] = df['value(t)']+2 #musée des horreurs (méthode pas générale du tout, du tout ...)

value = df['value(t)']
value = value.shift(-decalage)

df['value(t+{0})'.format(decalage)] = value
df=df[:-decalage]
df = df.round(0).astype(int)#why the type has been changed to floatby the method "shift"?

print(df)


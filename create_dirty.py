#load the table
df = pd.read_csv('data/taladrod.csv')

#fuzzy bkk
bkk_idx = list(df[df.contact_location=='กรุงเทพฯ'].index)
bkk_names = ['กรุงเทพ','กรุงเทพฯ','เมืองกรุงเทพฯ','กรุงเทพมหานคร','จังหวัดกรุงเทพฯ']
for i in bkk_idx:
    df.iloc[i,df.columns.get_loc('contact_location')] = np.random.choice(bkk_names, p = [0.11,0.53,0.09,0.14,0.13])
df.iloc[bkk_idx,:].contact_location.value_counts()

#add duplicates
df_dup = df.iloc[np.random.choice(range(int(df.shape[0]/2)),size=6000,replace=True),:]
df_dirty = pd.concat([df,df_dup]).sample(frac=1).reset_index(drop=True)
df_dirty.shape

df_dirty.to_csv('data/taladrod_dirty.csv',index=False)
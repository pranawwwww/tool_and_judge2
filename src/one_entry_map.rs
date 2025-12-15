use serde::de::{Error as DeError, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValuePair<K, V> {
    pub key: K,
    pub value: V,
}

impl<'de, K, V> Deserialize<'de> for KeyValuePair<K, V>
where
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OneEntryVisitor<K, V>(std::marker::PhantomData<(K, V)>);

        impl<'de, K, V> Visitor<'de> for OneEntryVisitor<K, V>
        where
            K: Deserialize<'de>,
            V: Deserialize<'de>,
        {
            type Value = KeyValuePair<K, V>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map with exactly one key-value pair")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let (key, value) = map
                    .next_entry()?
                    .ok_or_else(|| DeError::custom("expected exactly one entry, found none"))?;

                if map.next_entry::<K, V>()?.is_some() {
                    return Err(DeError::custom(
                        "expected exactly one entry, found more than one",
                    ));
                }

                Ok(KeyValuePair { key, value })
            }
        }

        deserializer.deserialize_map(OneEntryVisitor(std::marker::PhantomData))
    }
}

impl<K, V> Serialize for KeyValuePair<K, V>
where
    K: Serialize,
    V: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(&self.key, &self.value)?;
        map.end()
    }
}

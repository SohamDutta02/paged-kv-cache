//! Least-recently-used order over block ids: a doubly linked list threaded
//! through two index arrays, so push, remove and pop are O(1) and allocate
//! nothing after construction.

const NIL: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub(crate) struct Lru {
    prev: Vec<u32>,
    next: Vec<u32>,
    linked: Vec<bool>,
    /// Least recently used.
    head: u32,
    /// Most recently used.
    tail: u32,
    len: usize,
}

impl Lru {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            prev: vec![NIL; capacity],
            next: vec![NIL; capacity],
            linked: vec![false; capacity],
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn contains(&self, i: usize) -> bool {
        self.linked[i]
    }

    pub(crate) fn push_back(&mut self, i: usize) {
        debug_assert!(!self.linked[i], "{i} is already in the list");
        let i32 = i as u32;
        self.prev[i] = self.tail;
        self.next[i] = NIL;
        if self.tail == NIL {
            self.head = i32;
        } else {
            self.next[self.tail as usize] = i32;
        }
        self.tail = i32;
        self.linked[i] = true;
        self.len += 1;
    }

    pub(crate) fn remove(&mut self, i: usize) {
        debug_assert!(self.linked[i], "{i} is not in the list");
        let (prev, next) = (self.prev[i], self.next[i]);
        if prev == NIL {
            self.head = next;
        } else {
            self.next[prev as usize] = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.prev[next as usize] = prev;
        }
        self.prev[i] = NIL;
        self.next[i] = NIL;
        self.linked[i] = false;
        self.len -= 1;
    }

    pub(crate) fn pop_front(&mut self) -> Option<usize> {
        if self.head == NIL {
            return None;
        }
        let i = self.head as usize;
        self.remove(i);
        Some(i)
    }

    /// Least recently used first.
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut cursor = self.head;
        std::iter::from_fn(move || {
            if cursor == NIL {
                return None;
            }
            let i = cursor as usize;
            cursor = self.next[i];
            Some(i)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(lru: &Lru) -> Vec<usize> {
        lru.iter().collect()
    }

    #[test]
    fn pops_in_least_recently_used_order() {
        let mut lru = Lru::new(8);
        for i in [3, 1, 4] {
            lru.push_back(i);
        }
        assert_eq!(order(&lru), [3, 1, 4]);
        assert_eq!(lru.pop_front(), Some(3));
        assert_eq!(lru.pop_front(), Some(1));
        assert_eq!(lru.pop_front(), Some(4));
        assert_eq!(lru.pop_front(), None);
        assert_eq!(lru.len(), 0);
    }

    #[test]
    fn removes_from_head_middle_and_tail() {
        let mut lru = Lru::new(8);
        for i in 0..5 {
            lru.push_back(i);
        }
        lru.remove(2);
        lru.remove(0);
        lru.remove(4);
        assert_eq!(order(&lru), [1, 3]);
        assert!(!lru.contains(2));
        assert!(lru.contains(3));

        lru.push_back(0);
        assert_eq!(order(&lru), [1, 3, 0]);
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn matches_a_reference_model_under_random_operations() {
        let mut lru = Lru::new(16);
        let mut model: Vec<usize> = Vec::new();
        let mut seed: u64 = 7;
        for _ in 0..5_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let i = ((seed >> 33) % 16) as usize;
            match (seed >> 40) % 3 {
                0 if !lru.contains(i) => {
                    lru.push_back(i);
                    model.push(i);
                }
                1 if lru.contains(i) => {
                    lru.remove(i);
                    model.retain(|&x| x != i);
                }
                _ => {
                    let expected = (!model.is_empty()).then(|| model.remove(0));
                    assert_eq!(lru.pop_front(), expected);
                }
            }
            assert_eq!(order(&lru), model);
            assert_eq!(lru.len(), model.len());
        }
    }
}
